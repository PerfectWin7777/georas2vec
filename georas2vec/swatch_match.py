# Comparateur de cases appris : empreinte (couleur + motif) d'une case de légende et d'un morceau de polygone scanné.
# Entraînement : polygones de la feuille NORD ; test : feuille SUD (même légende USGS, scans différents).
import os, glob, pickle, random, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
import code_reader as CR
torch.set_num_threads(4)
PH, PW = 32, 64

# ---- cases de légende : image (texte noir effacé) + code vérifié
def legend_swatches():
    fn = "swatch_imgs.pkl"
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    import difflib
    rgb = np.asarray(Image.open("../dl_ds424/legend200-1.png").convert("RGB"))
    S = pd.read_pickle("../dl_ds424/swatches.pkl").reset_index(drop=True)
    V = pickle.load(open("code_vocab.pkl", "rb"))["gmna"]
    crops = [rgb[int(r.y0) + 2:int(r.y1) - 2, int(r.x0) + 2:int(r.x1) - 2] for r in S.itertuples()]
    reads = CR.Reader("code_reader.pt").read([Image.fromarray(c) for c in crops])
    out = {}
    for c, (code, p) in zip(crops, reads):
        code = code if code in V else (difflib.get_close_matches(code, V, n=1, cutoff=0.5) or [""])[0]
        if not code:
            continue
        a = c.copy(); dark = a.max(2) < 110
        a[ndi.binary_dilation(dark, iterations=2)] = np.median(a[~dark], 0) if (~dark).any() else 255   # efface le code imprimé
        if code not in out or p > out[code][1]:
            out[code] = (np.asarray(Image.fromarray(a).resize((PW, PH), Image.BILINEAR)), p)
    out = {k: v[0] for k, v in out.items()}
    pickle.dump(out, open(fn, "wb")); return out


# ---- morceaux de polygones scannés (vrai code connu)
def poly_patches(sheet, per_poly=3, min_area=1500, seed=0):
    fn = f"poly_patches_{sheet}.pkl"
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    rng = np.random.default_rng(seed); X, Y, A, Z = [], [], [], []
    for f in sorted(glob.glob(f"naming_cache/{sheet}_*.pkl")):
        R = pickle.load(open(f, "rb")); inst = R["inst"]; rgb = R["rgb"]; area = np.bincount(inst.ravel())
        for i in np.nonzero(area >= min_area)[0]:
            if i == 0:
                continue
            m = ndi.binary_erosion(inst == i, iterations=6)
            ys, xs = np.nonzero(m)
            if len(ys) < 50:
                continue
            for _ in range(per_poly):
                k = rng.integers(len(ys)); y, x = ys[k], xs[k]
                y0, x0 = max(0, y - PH), max(0, x - PW)
                patch = rgb[y0:y0 + 2 * PH, x0:x0 + 2 * PW]
                if patch.shape[0] < 2 * PH or patch.shape[1] < 2 * PW:
                    continue
                X.append(np.asarray(Image.fromarray(patch).resize((PW, PH), Image.BILINEAR))); Y.append(CR.canon(R["codes"][i - 1])); A.append(area[i]); Z.append(os.path.basename(f))
        print("patches", f, len(X), flush=True)
    d = dict(X=np.stack(X), Y=np.array(Y), A=np.array(A), Z=np.array(Z)); pickle.dump(d, open(fn, "wb")); return d


class Enc(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(c(6, 32), c(32, 32), nn.MaxPool2d(2), c(32, 64), c(64, 64), nn.MaxPool2d(2), c(64, 128), nn.AdaptiveAvgPool2d(1))
        self.h = nn.Linear(128 + 6, d)

    def forward(self, x):                        # x : B,3,H,W dans [0,1] -> RGB + Lab approx (moyennes/écarts) + carte
        mu, sd = x.mean((2, 3)), x.std((2, 3))
        z = self.f(torch.cat([x, x - mu[:, :, None, None]], 1)).flatten(1)
        return Fn.normalize(self.h(torch.cat([z, mu, sd], 1)), dim=1)


def swatch_image(crop):
    """Case de légende -> image normalisée (code imprimé effacé), PH x PW."""
    a = np.asarray(crop).copy(); dark = a.max(2) < 110
    if (~dark).any():
        a[ndi.binary_dilation(dark, iterations=2)] = np.median(a[~dark], 0)
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((PW, PH), Image.BILINEAR))


@torch.no_grad()
def embed_images(enc, imgs, bs=512):
    return np.concatenate([enc(tens(np.stack(imgs[s:s + bs]))).numpy() for s in range(0, len(imgs), bs)]) if len(imgs) else np.zeros((0, 96), np.float32)


def poly_embeddings(enc, rgb, inst, ids, n=3, seed=0):
    """Empreinte moyenne de quelques morceaux tirés dans chaque polygone."""
    rng = np.random.default_rng(seed); patches, owner = [], []
    order = np.argsort(inst.ravel(), kind="stable"); sl = inst.ravel()[order]
    for k, i in enumerate(ids):
        a, b = np.searchsorted(sl, i), np.searchsorted(sl, i, side="right")
        if b <= a:
            continue
        for j in rng.choice(order[a:b], min(n, b - a), replace=False):
            y, x = divmod(int(j), inst.shape[1])
            y0 = int(np.clip(y - PH, 0, max(0, inst.shape[0] - 2 * PH))); x0 = int(np.clip(x - PW, 0, max(0, inst.shape[1] - 2 * PW)))
            patches.append(np.asarray(Image.fromarray(rgb[y0:y0 + 2 * PH, x0:x0 + 2 * PW]).resize((PW, PH), Image.BILINEAR))); owner.append(k)
    E = embed_images(enc, patches); owner = np.array(owner); out = np.zeros((len(ids), E.shape[1] if len(E) else 96), np.float32)
    for k in range(len(ids)):
        m = owner == k
        if m.any():
            v = E[m].mean(0); out[k] = v / (np.linalg.norm(v) + 1e-9)
    return out


def load_encoder(path):
    enc = Enc(); enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); enc.eval(); return enc


def tens(a):
    return torch.from_numpy(np.asarray(a, np.float32) / 255.0).permute(0, 3, 1, 2)


def augment(x):
    x = x * (1 + 0.08 * torch.randn(x.shape[0], 3, 1, 1)) + 0.04 * torch.randn(x.shape[0], 3, 1, 1)
    return x.clamp(0, 1)


if __name__ == "__main__":
    SW = legend_swatches(); codes = sorted(SW); cidx = {c: i for i, c in enumerate(codes)}
    Simg = tens(np.stack([SW[c] for c in codes]))
    TR, TE = poly_patches("north"), poly_patches("south")
    keep = np.array([y in cidx for y in TR["Y"]]); Xtr, Ytr = TR["X"][keep], np.array([cidx[y] for y in TR["Y"][keep]])
    print("reprise au pas", start if 'start' in dir() else 1) if False else None
    print("légende", len(codes), "cases | entraînement", len(Xtr), "morceaux,", len(set(Ytr)), "unités", flush=True)
    enc = Enc(); opt = torch.optim.AdamW(enc.parameters(), 2e-3, weight_decay=1e-4); tau = 0.07
    start = 1
    if os.path.exists("swatch_ck.pt"):                       # reprise après redémarrage
        ck = torch.load("swatch_ck.pt", weights_only=False); enc.load_state_dict(ck["m"]); opt.load_state_dict(ck["o"]); start = ck["step"] + 1
    elif os.path.exists("swatch_enc.pt"):
        enc.load_state_dict(torch.load("swatch_enc.pt", weights_only=True)); start = 1001
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000, last_epoch=-1)
    for _ in range(start - 1):
        sched.step()
    t = time.time()
    for step in range(start, 3001):
        enc.train(); b = np.random.choice(len(Xtr), 128)
        xp = augment(tens(Xtr[b])); ys = Ytr[b]
        # toutes les cases de la légende comme classes (InfoNCE sur 850 cases) — cases augmentées légèrement
        es = enc(augment(Simg)); ep = enc(xp)
        loss = Fn.cross_entropy(ep @ es.T / tau, torch.from_numpy(ys))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0:
            enc.eval()
            with torch.no_grad():
                es = enc(Simg); r1 = r3 = tot = 0.0
                ok = np.array([y in cidx for y in TE["Y"]])
                Xte, Yte, Ate = TE["X"][ok], np.array([cidx[y] for y in TE["Y"][ok]]), TE["A"][ok]
                S_ = torch.cat([enc(tens(Xte[i:i + 512])) @ es.T for i in range(0, len(Xte), 512)])
                o = S_.argsort(1, descending=True).numpy()
                r1 = (Ate * (o[:, 0] == Yte)).sum() / Ate.sum(); r3 = (Ate * (o[:, :3] == Yte[:, None]).any(1)).sum() / Ate.sum()
            print(f"pas {step} perte {loss.item():.3f} | SUD : bonne case en 1er {r1:.1%}, dans les 3 premières {r3:.1%} ({time.time() - t:.0f}s)", flush=True)
            torch.save(enc.state_dict(), f"swatch_enc_{step}.pt"); torch.save(dict(m=enc.state_dict(), o=opt.state_dict(), step=step), "swatch_ck.pt")
            torch.save(enc.state_dict(), "swatch_enc.pt")
