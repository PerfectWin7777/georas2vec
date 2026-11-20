# Entraînement CPU du séparateur sur ../layers_ds (validation = split 'validation').
import os, io, sys, glob, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, "."); import separator as SP
torch.set_num_threads(4); torch.manual_seed(0)
D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("../layers_ds/*.parquet"))])
dec = lambda b, m="RGB": np.asarray(Image.open(io.BytesIO(b)).convert(m)) if m else np.asarray(Image.open(io.BytesIO(b)))
TR, VA = D[D.split == "train"].reset_index(drop=True), D[D.split == "validation"].reset_index(drop=True)
print("train", len(TR), "validation", len(VA), flush=True)
CROP, B, STEPS = 256, 8, int(os.environ.get("STEPS", 3000))
rng = np.random.default_rng(0)
W = torch.tensor([0.3, 3, 2, 4, 2, 2, 2.0])       # classes rares plus lourdes


def batch():
    xs, ps, ks = [], [], []
    for _ in range(B):
        r = TR.iloc[rng.integers(len(TR))]
        x, p, k = dec(r.image), dec(r.plate), dec(r.ink, None)
        i, j = rng.integers(0, 512 - CROP + 1, 2); sl = (slice(i, i + CROP), slice(j, j + CROP))
        x, p, k = x[sl], p[sl], k[sl]
        if rng.random() < 0.5:
            x, p, k = x[:, ::-1], p[:, ::-1], k[:, ::-1]
        xs.append(x.transpose(2, 0, 1)); ps.append(p.transpose(2, 0, 1)); ks.append(k)
    return (torch.from_numpy(np.ascontiguousarray(xs)), torch.from_numpy(np.ascontiguousarray(ps)).float() / 255,
            torch.from_numpy(np.ascontiguousarray(ks)).long())


m = SP.Separator(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS, pct_start=0.05)
t0, s0 = time.time(), 0
if os.path.exists("separator_ck.pt"):            # reprise après redémarrage
    ck = torch.load("separator_ck.pt", weights_only=False); m.load_state_dict(ck["m"]); opt.load_state_dict(ck["o"]); sch.load_state_dict(ck["s"])
    s0 = ck["step"]; rng = np.random.default_rng(s0); print("reprise au pas", s0, flush=True)
for s in range(s0 + 1, STEPS + 1):
    x, p, k = batch(); m.train()
    pl, lg = m(x)
    wpix = 1 + 9 * (k > 0).float()[:, None]
    loss = (torch.abs(pl - p) * wpix).mean() * 10 + F.cross_entropy(lg, k, weight=W)
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s % 100 == 0:
        print(f"pas {s} perte {loss.item():.3f} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    if s % 200 == 0 or s == STEPS:
        torch.save({"m": m.state_dict(), "o": opt.state_dict(), "s": sch.state_dict(), "step": s}, "separator_ck.pt.tmp")
        os.replace("separator_ck.pt.tmp", "separator_ck.pt"); torch.save(m.state_dict(), "separator.pt")
# validation : erreur de plaque sur l'encre (vs image brute) et IoU par classe
conf = np.zeros((SP.NCLS, SP.NCLS), np.int64); e_in, e_raw = [], []
for r in VA.itertuples():
    x, p, k = dec(r.image), dec(r.plate), dec(r.ink, None)
    pl, pr = SP.separate(m, x); pk = pr.argmax(0)
    conf += np.bincount(k.ravel() * SP.NCLS + pk.ravel(), minlength=SP.NCLS ** 2).reshape(SP.NCLS, SP.NCLS)
    on = k > 0
    if on.any():
        e_in.append(np.abs(pl.astype(int) - p)[on].mean()); e_raw.append(np.abs(x.astype(int) - p)[on].mean())
iou = np.diag(conf) / np.maximum(conf.sum(0) + conf.sum(1) - np.diag(conf), 1)
print("erreur plaque sur l'encre (0-255) : brute %.1f -> séparateur %.1f" % (np.mean(e_raw), np.mean(e_in)))
print("IoU par classe :", {c: round(float(v), 3) for c, v in zip(["aucune", "contact", "côte", "faille", "texte", "autre", "topo"], iou)})
print("FIN_SEP", flush=True)
