# =====================================================================================
#  GEORAS2VEC – ENTRAÎNEMENT FINAL
#  Modèle retenu par le labo (rounds 1-3) : U-Net++ / EfficientNet-B4, entrée RGB+Lab,
#  augmentations de couleur, affinités (10 décalages) + failles, mutex watershed.
#  Données : dataset GeoRas2Vec complet (6 600 tuiles, 4 continents), split spatial.
#  Sorties (/kaggle/working/georas2vec_model) :
#    best.pt (poids + config), best_fp16.pt, history.csv, test_metrics.csv, metrics.json, figures
# =====================================================================================
import os, io, sys, json, time, copy
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage as ndi
from skimage.metrics import adapted_rand_error, variation_of_information
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from numba import njit

E = os.environ.get
STEPS = int(E("GC_STEPS", 30000))            # pas d'entraînement
BATCH, CROP, LR = int(E("GC_BATCH", 8)), int(E("GC_CROP", 448)), float(E("GC_LR", 3e-4))
EVAL_EVERY = int(E("GC_EVAL_EVERY", 2500))
N_VAL = int(E("GC_NVAL", 12))                # tuiles de validation par source (suivi pendant l'entraînement)
P_REAL = float(E("GC_PREAL", 0.35))          # part de tuiles RÉELLES dans chaque lot (elles ne font que 21 % du dataset)
MAX_HOURS = float(E("GC_MAX_HOURS", 8.0))    # garde-fou : on arrête l'entraînement avant la limite Kaggle (12 h)
N_TEST = int(E("GC_NTEST", 0))               # 0 = tout le jeu de test
OUT = Path(E("GC_OUT", "/kaggle/working/georas2vec_model")); OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T_START = time.time()
print("Matériel :", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)

CFG = dict(arch="unetpp", encoder=E("GC_ENCODER", "efficientnet-b4"), input="rgb+lab",
           offsets=[(1, 0), (0, 1), (3, 0), (0, 3), (9, 0), (0, 9), (9, 9), (9, -9), (27, 0), (0, 27)],
           n_short=2, mws_stride=int(E("GC_MWS_STRIDE", 1)), mws_min_size=int(E("GC_MINSIZE", 40)), tile=512,
           small_w=E("GC_SMALLW", "0") == "1", ohem=E("GC_OHEM", "0") == "1", tta=E("GC_TTA", "0") == "1")
OFFSETS, N_SHORT, K = CFG["offsets"], CFG["n_short"], len(CFG["offsets"])


# ------------------------------------------------------------------ données
def find_dir(marker):
    root = Path(E("GC_INPUT", "/kaggle/input"))
    hits = sorted(root.rglob(marker))
    return hits[0].parent if hits else None


DATA = Path(E("GC_DATA")) if E("GC_DATA") else find_dir("test-00000.parquet")       # dataset complet
HARD = Path(E("GC_HARD")) if E("GC_HARD") else find_dir("hard_mono-00000.parquet")  # jeux pièges (georas2vec-mini)
assert DATA is not None, "dataset GeoRas2Vec introuvable (ajoute la sortie du notebook georas2vec-dataset en entrée)"
print("Données :", DATA, "| pièges :", HARD, flush=True)


def read_split(folder, split):
    """Garde les PNG compressés en mémoire (2 Go au lieu de 11 Go décodés) ; décodage à la volée."""
    rows = []
    for f in sorted(Path(folder).glob(f"{split}-*.parquet")):
        t = pq.read_table(f, columns=["id", "source", "continent", "kind", "image", "instances", "faults"]).to_pylist()
        rows += [dict(id=r["id"], source=r["source"], continent=r["continent"], kind=r["kind"],
                      image=r["image"]["bytes"], inst=r["instances"]["bytes"], faults=r["faults"]["bytes"]) for r in t]
    return rows


def decode(r):
    rgb = np.asarray(Image.open(io.BytesIO(r["image"])).convert("RGB"))
    inst = np.asarray(Image.open(io.BytesIO(r["inst"]))).astype(np.int32)
    flt = (np.asarray(Image.open(io.BytesIO(r["faults"]))) > 0).astype(np.uint8)
    return rgb, inst, flt


TRAIN = read_split(DATA, "train")
VAL_ALL = read_split(DATA, "validation")
TEST = read_split(DATA, "test")
HARDSETS = {k: read_split(HARD, k) for k in ("hard_mono", "hard_pattern")} if HARD else {}
REAL_IDX = [i for i, r in enumerate(TRAIN) if r["kind"] == "real"]
SYN_IDX = [i for i, r in enumerate(TRAIN) if r["kind"] != "real"]


def spread(rows, n):
    return [rows[i] for i in np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)] if rows else []


VAL = {}   # suivi : n tuiles par (source, type), réparties régulièrement
for key in sorted({(r["source"], r["kind"]) for r in VAL_ALL}):
    VAL[key] = spread([r for r in VAL_ALL if (r["source"], r["kind"]) == key], N_VAL * (3 if key[1] == "real" else 1))
print(f"train {len(TRAIN)} ({len(REAL_IDX)} réelles) | validation suivie {sum(map(len, VAL.values()))} | test {len(TEST)} | "
      + " | ".join(f"{k} {len(v)}" for k, v in HARDSETS.items()), flush=True)


# ------------------------------------------------------------------ affinités + mutex watershed
def fill_holes(inst):
    """Chaque zone sans polygone (mer, lac, halo) devient sa propre région."""
    holes, n = ndi.label(inst == 0)
    if n == 0:
        return inst
    out = inst.copy()
    out[holes > 0] = holes[holes > 0] + inst.max()
    return out


def affinity_targets(inst):
    H, W = inst.shape
    aff, mask = np.zeros((K, H, W), np.float32), np.zeros((K, H, W), np.float32)
    for k, (dy, dx) in enumerate(OFFSETS):
        ys, ye, xs, xe = max(0, -dy), H - max(0, dy), max(0, -dx), W - max(0, dx)
        a, b = inst[ys:ye, xs:xe], inst[ys + dy:ye + dy, xs + dx:xe + dx]
        aff[k, ys:ye, xs:xe] = a == b
        mask[k, ys:ye, xs:xe] = 1          # après fill_holes, tous les pixels ont une région
    return aff, mask


@njit
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit
def _mws_core(n, u, v, rep):
    parent = np.arange(n); size = np.ones(n, np.int64); m = len(u)
    mutex_a = np.empty(m, np.int64); mutex_b = np.empty(m, np.int64); nm = 0
    head = -np.ones(n, np.int64); nxt = np.empty(2 * m, np.int64); val = np.empty(2 * m, np.int64); nl = 0
    for e in range(m):
        ra, rb = _find(parent, u[e]), _find(parent, v[e])
        if ra == rb:
            continue
        if rep[e]:
            val[nl] = nm; nxt[nl] = head[ra]; head[ra] = nl; nl += 1
            val[nl] = nm; nxt[nl] = head[rb]; head[rb] = nl; nl += 1
            mutex_a[nm] = ra; mutex_b[nm] = rb; nm += 1
        else:
            small, other = (ra, rb) if size[ra] <= size[rb] else (rb, ra)
            blocked = False; p = head[small]
            while p != -1:
                k = val[p]; x = _find(parent, mutex_a[k]); y = _find(parent, mutex_b[k])
                if (x == small and y == other) or (x == other and y == small):
                    blocked = True; break
                p = nxt[p]
            if blocked:
                continue
            parent[small] = other; size[other] += size[small]; p = head[small]
            if p != -1:
                while nxt[p] != -1:
                    p = nxt[p]
                nxt[p] = head[other]; head[other] = head[small]
    out = np.empty(n, np.int64)
    for i in range(n):
        out[i] = _find(parent, i)
    return out


def mutex_watershed(aff, stride=CFG["mws_stride"], min_size=CFG["mws_min_size"]):
    _, H, W = aff.shape
    idx = np.arange(H * W).reshape(H, W)
    us, vs, ws, rep = [], [], [], []
    for k, (dy, dx) in enumerate(OFFSETS):
        a = idx[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        b = idx[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
        w = aff[k, max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)].copy()
        if k >= N_SHORT:
            a, b, w = a[::stride, ::stride], b[::stride, ::stride], 1 - w[::stride, ::stride]
        us.append(a.ravel()); vs.append(b.ravel()); ws.append(w.ravel()); rep.append(np.full(a.size, k >= N_SHORT))
    u, v, w, rep = map(np.concatenate, (us, vs, ws, rep))
    order = np.argsort(-w, kind="stable")
    lab = _mws_core(H * W, u[order], v[order], rep[order]).reshape(H, W)
    _, lab = np.unique(lab, return_inverse=True); lab = lab.reshape(H, W) + 1
    sizes = np.bincount(lab.ravel()); tiny = sizes < min_size; tiny[0] = False
    if tiny[lab].any():
        _, (iy, ix) = ndi.distance_transform_edt(tiny[lab], return_indices=True)
        lab = lab[iy, ix]; _, lab = np.unique(lab, return_inverse=True); lab = lab.reshape(H, W) + 1
    return lab.astype(np.int32)


# ------------------------------------------------------------------ augmentations
def color_aug(rgb, rng):
    """Casse le lien 'couleur = unité' : rotation de teinte, saturation, gains, permutation, gris, contraste, bruit."""
    x = rgb.astype(np.float32) / 255
    if rng.random() < 0.8:
        T = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)
        a = rng.uniform(-np.pi, np.pi)
        Rm = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], np.float32)
        Rm[1:, 1:] *= rng.uniform(0.6, 1.3)
        x = x @ (np.linalg.inv(T) @ Rm @ T).T
    if rng.random() < 0.3:
        x = x[..., rng.permutation(3)]
    x = x * rng.uniform(0.85, 1.15, 3) + rng.uniform(-0.06, 0.06, 3)
    if rng.random() < 0.1:
        x = np.repeat(x.mean(-1, keepdims=True), 3, -1)
    m = x.mean()
    x = (x - m) * rng.uniform(0.75, 1.2) + m
    if rng.random() < 0.3:
        x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


class TrainSet(Dataset):
    def __len__(self):
        return STEPS * BATCH

    def __getitem__(self, i):
        rng = np.random.default_rng([i, 20261112])
        pool = REAL_IDX if (REAL_IDX and rng.random() < P_REAL) else SYN_IDX
        rgb, inst, flt = decode(TRAIN[pool[rng.integers(len(pool))]])
        inst = fill_holes(inst)                                   # sur la tuile entière, avant recadrage
        y, x = rng.integers(0, rgb.shape[0] - CROP + 1), rng.integers(0, rgb.shape[1] - CROP + 1)
        rgb, inst, flt = (a[y:y + CROP, x:x + CROP] for a in (rgb, inst, flt))
        k = int(rng.integers(4))
        rgb, inst, flt = (np.rot90(a, k) for a in (rgb, inst, flt))
        if rng.random() < 0.5:
            rgb, inst, flt = rgb[:, ::-1], inst[:, ::-1], flt[:, ::-1]
        rgb = color_aug(rgb, rng)
        inst = np.ascontiguousarray(inst)
        aff, mask = affinity_targets(inst)
        if CFG["small_w"]:     # les arêtes touchant un petit polygone pèsent plus (jusqu'à x8)
            area = np.bincount(inst.ravel())
            w = np.clip(np.sqrt(2000.0 / np.maximum(area[inst], 1)), 1.0, 8.0).astype(np.float32)
            H, W = inst.shape
            for k, (dy, dx) in enumerate(OFFSETS):
                ys, ye, xs, xe = max(0, -dy), H - max(0, dy), max(0, -dx), W - max(0, dx)
                mask[k, ys:ye, xs:xe] = np.maximum(w[ys:ye, xs:xe], w[ys + dy:ye + dy, xs + dx:xe + dx])
        return (torch.from_numpy(np.ascontiguousarray(rgb).transpose(2, 0, 1).copy()), torch.from_numpy(aff),
                torch.from_numpy(mask), torch.from_numpy(np.ascontiguousarray(flt, np.float32))[None])


# ------------------------------------------------------------------ modèle
def rgb2lab(x):
    """x : B x 3 x H x W dans [0,1] -> Lab normalisé (L/50-1, a/64, b/64)."""
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], device=x.device)
    xyz = torch.einsum("ij,bjhw->bihw", M, lin) / torch.tensor([0.95047, 1.0, 1.08883], device=x.device).view(1, 3, 1, 1)
    f = torch.where(xyz > 0.008856, xyz.clamp(min=1e-6) ** (1 / 3), 7.787 * xyz + 16 / 116)
    L, a, b = 116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L / 50 - 1, a / 64, b / 64], 1)


class GeoRas2VecNet(nn.Module):
    """Entrée : image uint8 B x 3 x H x W (H, W multiples de 32). Sortie : K affinités + 1 carte de failles (logits)."""

    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        Arch = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus}[cfg["arch"]]
        self.net = Arch(cfg["encoder"], encoder_weights="imagenet" if pretrained else None, in_channels=6,
                        classes=len(cfg["offsets"]) + 1)

    def forward(self, x8):
        with torch.autocast(x8.device.type, enabled=False):      # conversion de couleur en précision complète
            x = x8.float() / 255
            x = torch.cat([(x - self.mean) / self.std, rgb2lab(x)], 1)
        return self.net(x)


def dice(p, t, eps=1.0, dims=None):
    return 1 - (2 * (p * t).sum(dims) + eps) / (p.sum(dims) + t.sum(dims) + eps)


def loss_fn(out, aff, mask, flt):
    bce = F.binary_cross_entropy_with_logits(out[:, :K], aff, reduction="none") * mask
    la = bce.sum() / mask.sum().clamp(min=1)
    if CFG["ohem"]:         # + moyenne des 5 % d'arêtes les plus mal prédites
        v = bce.flatten()
        la = la + v.topk(max(1, v.numel() // 20)).values.mean() / mask.mean().clamp(min=1e-6)
    p, t = (1 - torch.sigmoid(out[:, :K])) * mask, (1 - aff) * mask          # Dice sur les COUPURES (classe rare)
    la = la + dice(p, t, dims=(0, 2, 3)).mean()
    lf = F.binary_cross_entropy_with_logits(out[:, K:], flt, pos_weight=torch.tensor(10.0, device=out.device)) \
        + dice(torch.sigmoid(out[:, K:]), flt)
    return la + 0.5 * lf


# ------------------------------------------------------------------ évaluation
@torch.no_grad()
def predict(model, rgb):
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(DEV)
    with torch.autocast(DEV, enabled=DEV == "cuda"):
        o = model(x)
    o = torch.sigmoid(o.float())[0].cpu().numpy()
    return o[:K], o[K]


def predict_tta(model, rgb):
    """Moyenne sur 4 miroirs ; chaque décalage ramené dans le repère d'origine (testé : exact)."""
    H, W = rgb.shape[:2]
    Y, X = np.mgrid[:H, :W]
    idx = {tuple(o): k for k, o in enumerate(OFFSETS)}
    acc, accf = 0, 0
    for fy in (0, 1):
        for fx in (0, 1):
            im = rgb[::-1] if fy else rgb
            im = im[:, ::-1] if fx else im
            A, pf = predict(model, np.ascontiguousarray(im))
            py, px = (H - 1 - Y) if fy else Y, (W - 1 - X) if fx else X
            out = np.empty_like(A)
            for k, (dy, dx) in enumerate(OFFSETS):
                t = (-dy if fy else dy, -dx if fx else dx)
                if t in idx:
                    out[k] = A[idx[t]][py, px]
                else:
                    out[k] = A[idx[(-t[0], -t[1])]][np.clip(py + t[0], 0, H - 1), np.clip(px + t[1], 0, W - 1)]
            acc = acc + out; accf = accf + pf[py, px]
    return acc / 4, accf / 4


def tile_metrics(pred, pf, inst, flt):
    gt = fill_holes(inst)
    are = adapted_rand_error(gt, pred)[0]
    split, merge = variation_of_information(gt, pred)     # H(pred|vérité) = sur-découpage ; H(vérité|pred) = fusion
    small = [0, 0]; big = [0, 0]
    ids, area = np.unique(inst[inst > 0], return_counts=True)
    if len(ids):
        # IoU de chaque instance avec la région prédite qui la recouvre le plus
        pair = inst.astype(np.int64) * (pred.max() + 1) + pred
        pv, pc = np.unique(pair[inst > 0], return_counts=True)
        gi, pi = pv // (pred.max() + 1), pv % (pred.max() + 1)
        parea = np.bincount(pred.ravel())
        order = np.lexsort((-pc, gi))
        first = np.r_[True, gi[order][1:] != gi[order][:-1]]
        g, inter, pa = gi[order][first], pc[order][first], parea[pi[order][first]]
        a = area[np.searchsorted(ids, g)]
        ok = inter / (a + pa - inter) >= 0.5
        small = [int(ok[a < 1000].sum()), int((a < 1000).sum())]; big = [int(ok[a >= 1000].sum()), int((a >= 1000).sum())]
        mid = [int(ok[(a >= 200) & (a < 1000)].sum()), int(((a >= 200) & (a < 1000)).sum())]
        tiny = [int(ok[a < 200].sum()), int((a < 200).sum())]
        surf = [float((a * ok).sum()), float(a.sum())]
    else:
        mid = tiny = [0, 0]; surf = [0.0, 0.0]
    pb, gb = pf > 0.5, flt > 0
    gd, pd_ = ndi.binary_dilation(gb, iterations=2), ndi.binary_dilation(pb, iterations=2)
    return dict(ARAND=are, split=split, merge=merge, s_ok=small[0], s_n=small[1], b_ok=big[0], b_n=big[1],
                t_ok=tiny[0], t_n=tiny[1], m_ok=mid[0], m_n=mid[1], surf_ok=surf[0], surf_n=surf[1],
                tp=int((pb & gd).sum()), fp=int((pb & ~gd).sum()), fn=int((gb & ~pd_).sum()))


def summarize(rows):
    d = pd.DataFrame(rows)
    f1 = 2 * d.tp.sum() / max(2 * d.tp.sum() + d.fp.sum() + d.fn.sum(), 1) if (d.tp.sum() + d.fn.sum()) > 0 else np.nan
    return dict(tuiles=len(d), ARAND=d.ARAND.mean(), sur_decoupage=d.split.mean(), fusion=d["merge"].mean(),
                rappel_petites=d.s_ok.sum() / max(d.s_n.sum(), 1), rappel_grandes=d.b_ok.sum() / max(d.b_n.sum(), 1),
                rappel_40_200=d.t_ok.sum() / max(d.t_n.sum(), 1), rappel_200_1000=d.m_ok.sum() / max(d.m_n.sum(), 1),
                rappel_tous=(d.s_ok.sum() + d.b_ok.sum()) / max(d.s_n.sum() + d.b_n.sum(), 1),
                surface_correcte=d.surf_ok.sum() / max(d.surf_n.sum(), 1), F1_failles=f1)


def evaluate(model, rows, tta=False):
    model.eval()
    out = []
    for r in rows:
        rgb, inst, flt = decode(r)
        aff, pf = predict_tta(model, rgb) if tta else predict(model, rgb)
        out.append(tile_metrics(mutex_watershed(aff), pf, inst, flt))
    return out


def val_score(model):
    """Score de suivi : moyenne de l'ARAND réel et de l'ARAND synthétique (toutes sources à poids égal)."""
    res = {k: summarize(evaluate(model, v)) for k, v in VAL.items()}
    real = np.mean([v["ARAND"] for k, v in res.items() if k[1] == "real"])
    syn = np.mean([v["ARAND"] for k, v in res.items() if k[1] != "real"])
    return 0.5 * (real + syn), real, syn, res


# ------------------------------------------------------------------ entraînement
torch.manual_seed(0)
model = GeoRas2VecNet(CFG).to(DEV)
ema = copy.deepcopy(model).eval()                      # moyenne glissante des poids (souvent plus stable)
for p in ema.parameters():
    p.requires_grad_(False)
EMA_DECAY = 0.999
dl = DataLoader(TrainSet(), BATCH, shuffle=False, num_workers=min(4, os.cpu_count() or 1), pin_memory=DEV == "cuda",
                drop_last=True)
opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)
scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
mutex_watershed(np.random.rand(K, 64, 64).astype(np.float32))     # compilation numba
print(f"Paramètres : {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M | {STEPS} pas, lot {BATCH}, recadrage {CROP}", flush=True)

HIST, best, t0, run_loss, n_loss = [], 1e9, time.time(), 0.0, 0
model.train()
step = 0
for step, batch in enumerate(dl, 1):
    x, aff, mask, flt = (t.to(DEV, non_blocking=True) for t in batch)
    with torch.autocast(DEV, enabled=DEV == "cuda"):
        out = model(x)
    loss = loss_fn(out.float(), aff, mask, flt)
    if not torch.isfinite(loss):
        print(f"  pas {step} : perte non finie, lot ignoré", flush=True)
        opt.zero_grad(set_to_none=True); sched.step()
        continue
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(opt); scaler.update(); sched.step()
    with torch.no_grad():
        d = min(EMA_DECAY, (1 + step) / (10 + step))
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)
    run_loss += loss.item(); n_loss += 1
    if step % 250 == 0:
        print(f"pas {step}/{STEPS} | perte {run_loss / n_loss:.4f} | {(time.time() - t0) / 60:.0f} min", flush=True)
    hours = (time.time() - T_START) / 3600
    last = step == STEPS or hours > MAX_HOURS
    if step % EVAL_EVERY == 0 or last:
        row = dict(step=step, loss=run_loss / max(n_loss, 1), heures=round(hours, 2))
        for tag, m in (("modele", model), ("ema", ema)):
            sc, real, syn, _ = val_score(m)
            row.update({f"{tag}_score": sc, f"{tag}_reel": real, f"{tag}_synth": syn})
            if sc < best:
                best = sc
                torch.save({"state_dict": m.state_dict(), "cfg": CFG, "step": step, "val_score": float(sc), "poids": tag}, OUT / "best.pt")
                row["meilleur"] = tag
        model.train()
        HIST.append(row); pd.DataFrame(HIST).to_csv(OUT / "history.csv", index=False)
        print("  VALIDATION", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        run_loss, n_loss = 0.0, 0
    if last:
        if step < STEPS:
            print(f"Arrêt au pas {step} : limite de {MAX_HOURS} h atteinte.", flush=True)
        break
del dl

# ------------------------------------------------------------------ test final (meilleur point de contrôle)
ck = torch.load(OUT / "best.pt", map_location=DEV)
model = GeoRas2VecNet(CFG, pretrained=False).to(DEV)
model.load_state_dict(ck["state_dict"]); model.eval()
torch.save({"state_dict": {k: (v.half() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()},
            "cfg": CFG, "step": ck["step"]}, OUT / "best_fp16.pt")
print(f"\nMeilleur modèle : pas {ck['step']} ({ck['poids']}), score de validation {ck['val_score']:.4f}", flush=True)

test_rows = TEST if not N_TEST else spread(TEST, N_TEST)
res = evaluate(model, test_rows, tta=CFG["tta"])
T = pd.DataFrame(res).assign(continent=[r["continent"] for r in test_rows], source=[r["source"] for r in test_rows],
                             type=[r["kind"] for r in test_rows])
tables = []
for keys in (["continent"], ["source", "type"]):
    for name, g in T.groupby(keys):
        name = name if isinstance(name, tuple) else (name,)
        tables.append({"groupe": " / ".join(name), **summarize(g.to_dict("records"))})
tables.append({"groupe": "TEST COMPLET", **summarize(res)})
for k, rows in HARDSETS.items():
    rows = rows if not N_TEST else spread(rows, N_TEST)
    tables.append({"groupe": {"hard_mono": "PIÈGE unités de même couleur", "hard_pattern": "PIÈGE unités à motifs"}[k],
                   **summarize(evaluate(model, rows, tta=CFG["tta"]))})
TM = pd.DataFrame(tables).round(4)
TM.to_csv(OUT / "test_metrics.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("\n=== RÉSULTATS SUR LE JEU DE TEST (jamais vu à l'entraînement) ==="); print(TM.to_string(index=False), flush=True)
json.dump({"cfg": CFG, "steps": int(ck["step"]), "val_score": float(ck["val_score"]), "test": TM.to_dict("records"),
           "train_hours": round((time.time() - T_START) / 3600, 2)}, open(OUT / "metrics.json", "w"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
picks = [("réel", spread([r for r in TEST if r["kind"] == "real"], 4))]
picks += [(s, spread([r for r in TEST if r["source"] == s and r["kind"] != "real"], 2)) for s in sorted({r["source"] for r in TEST})]
picks += [(k, spread(v, 3)) for k, v in HARDSETS.items()]
rows_ = [(n, r) for n, rs in picks for r in rs]
fig, ax = plt.subplots(3, len(rows_), figsize=(4 * len(rows_), 12))
for j, (n, r) in enumerate(rows_):
    rgb, inst, flt = decode(r); aff, pf = predict(model, rgb); pred = mutex_watershed(aff)
    ax[0, j].imshow(rgb); ax[0, j].set_title(f"{n}\n{r['source']}", fontsize=9)
    ax[1, j].imshow(fill_holes(inst) % 20, cmap="tab20", interpolation="nearest"); ax[1, j].set_title("vérité", fontsize=9)
    ax[2, j].imshow(pred % 20, cmap="tab20", interpolation="nearest")
    ax[2, j].imshow(np.ma.masked_less(pf, 0.5), cmap="gray_r", interpolation="nearest"); ax[2, j].set_title("modèle", fontsize=9)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "test_examples.png", dpi=55); plt.close()
print("Fichiers :", sorted(p.name for p in OUT.iterdir()), f"| durée totale {(time.time() - T_START) / 3600:.2f} h", flush=True)
