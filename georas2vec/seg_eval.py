# Variantes de post-traitement du découpage, sur sorties en cache. Réglage sur VALIDATION, confirmation sur TEST.
#   python seg_eval.py <cache_dir> <split> <variante1,variante2,...>
import sys, os, glob, numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.metrics import adapted_rand_error, variation_of_information
import skimage.metrics as _skm
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gc_model as M
from numba import njit

OFF, NS = M.OFFSETS, M.N_SHORT


def mws(aff, min_size=40, beta=1.0, sigma=0.0):
    if sigma > 0:
        aff = np.stack([ndi.gaussian_filter(a, sigma) for a in aff])
    _, H, W = aff.shape
    idx = np.arange(H * W).reshape(H, W)
    us, vs, ws, rep = [], [], [], []
    for k, (dy, dx) in enumerate(OFF):
        a = idx[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        b = idx[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
        w = aff[k, max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)].copy()
        if k >= NS:
            w = np.clip((1 - w) * beta, 0, 1)
        us.append(a.ravel()); vs.append(b.ravel()); ws.append(w.ravel()); rep.append(np.full(a.size, k >= NS))
    u, v, w, rep = map(np.concatenate, (us, vs, ws, rep))
    order = np.argsort(-w, kind="stable")
    lab = M._mws_core(H * W, u[order], v[order], rep[order]).reshape(H, W)
    _, lab = np.unique(lab, return_inverse=True); lab = lab.reshape(H, W) + 1
    return remove_small(lab, min_size)


def remove_small(lab, min_size):
    sizes = np.bincount(lab.ravel()); tiny = sizes < min_size; tiny[0] = False
    if tiny[lab].any():
        _, (iy, ix) = ndi.distance_transform_edt(tiny[lab], return_indices=True)
        lab = lab[iy, ix]
    _, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(lab.shape) + 1


def mode_filter(lab, r=1, passes=1):
    """Lissage des contours : chaque pixel prend l'étiquette majoritaire de son voisinage (2r+1)²."""
    for _ in range(passes):
        H, W = lab.shape; best = lab.copy(); cnt = np.zeros(lab.shape, np.int16)
        P = np.pad(lab, r, mode="edge")
        shifts = [P[r + dy:r + dy + H, r + dx:r + dx + W] for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        for s in shifts:
            c = sum((s == t).astype(np.int16) for t in shifts)
            m = c > cnt; best[m] = s[m]; cnt[m] = c[m]
        lab = best
    return lab


def boundary_f(gt, pred, tol=1):
    bg = np.zeros(gt.shape, bool); bg[:, 1:] |= gt[:, 1:] != gt[:, :-1]; bg[1:] |= gt[1:] != gt[:-1]
    bp = np.zeros(gt.shape, bool); bp[:, 1:] |= pred[:, 1:] != pred[:, :-1]; bp[1:] |= pred[1:] != pred[:-1]
    dg = ndi.binary_dilation(bg, iterations=tol); dp = ndi.binary_dilation(bp, iterations=tol)
    p = (bp & dg).sum() / max(bp.sum(), 1); r = (bg & dp).sum() / max(bg.sum(), 1)
    return 2 * p * r / max(p + r, 1e-9)


def metrics(pred, pf, inst, flt, fthr=0.5):
    gt = M.fill_holes(inst)
    are = adapted_rand_error(gt, pred)[0]; split, merge = variation_of_information(gt, pred)
    ids, area = np.unique(inst[inst > 0], return_counts=True)
    r = dict(ARAND=are, split=split, merge=merge, bF=boundary_f(gt, pred))
    if len(ids):
        pair = inst.astype(np.int64) * (pred.max() + 1) + pred
        pv, pc = np.unique(pair[inst > 0], return_counts=True)
        gi, pi = pv // (pred.max() + 1), pv % (pred.max() + 1)
        parea = np.bincount(pred.ravel())
        order = np.lexsort((-pc, gi)); first = np.r_[True, gi[order][1:] != gi[order][:-1]]
        g, inter, pa = gi[order][first], pc[order][first], parea[pi[order][first]]
        a = area[np.searchsorted(ids, g)]; ok = inter / (a + pa - inter) >= 0.5
        r.update(s_ok=ok[a < 1000].sum(), s_n=(a < 1000).sum(), b_ok=ok[a >= 1000].sum(), b_n=(a >= 1000).sum(),
                 t_ok=ok[a < 200].sum(), t_n=(a < 200).sum(), surf_ok=(a * ok).sum(), surf_n=a.sum())
    pb, gb = pf > fthr, flt > 0
    gd, pd_ = ndi.binary_dilation(gb, iterations=2), ndi.binary_dilation(pb, iterations=2)
    r.update(tp=int((pb & gd).sum()), fp=int((pb & ~gd).sum()), fn=int((gb & ~pd_).sum()))
    return r


VARIANTS = {
    "base": dict(),
    "min25": dict(min_size=25), "min60": dict(min_size=60),
    "beta0.8": dict(beta=0.8), "beta0.9": dict(beta=0.9), "beta1.1": dict(beta=1.1), "beta1.25": dict(beta=1.25),
    "sig0.7": dict(sigma=0.7), "sig1.0": dict(sigma=1.0),
    "mode1": dict(mode=1), "mode2": dict(mode=2),
    "f0.3": dict(fthr=0.3), "f0.2": dict(fthr=0.2), "f0.25": dict(fthr=0.25),
    "min15": dict(min_size=15), "min20": dict(min_size=20), "f0.15": dict(fthr=0.15),
    "m20b1.03": dict(min_size=20, beta=1.03), "m20f0.2": dict(min_size=20, fthr=0.2), "m20b1.03f0.2": dict(min_size=20, beta=1.03, fthr=0.2),
    "beta1.03": dict(beta=1.03), "beta1.05": dict(beta=1.05),
    "m25b1.05": dict(min_size=25, beta=1.05), "m25b1.1": dict(min_size=25, beta=1.1), "m20b1.05": dict(min_size=20, beta=1.05), "f0.4": dict(fthr=0.4), "f0.6": dict(fthr=0.6), "f0.7": dict(fthr=0.7),
}


def run_one(args):
    fn, names = args
    z = np.load(fn)
    aff, pf = z["aff"].astype(np.float32) / 255, z["pf"].astype(np.float32) / 255
    inst, flt = z["inst"].astype(np.int32), z["flt"]
    out = []; cache = {}
    for nm in names:
        v = dict(VARIANTS[nm]); fthr = v.pop("fthr", 0.5); mode = v.pop("mode", 0)
        key = tuple(sorted(v.items()))
        if key not in cache:
            cache[key] = mws(aff, **v)
        pred = cache[key]
        if mode:
            pred = mode_filter(pred, r=mode)
        out.append(dict(variante=nm, fichier=os.path.basename(fn), source=str(z["source"]), kind=str(z["kind"]),
                        **metrics(pred, pf, inst, flt, fthr)))
    return out


def summarize(d):
    f1 = 2 * d.tp.sum() / max(2 * d.tp.sum() + d.fp.sum() + d.fn.sum(), 1) if (d.tp.sum() + d.fn.sum()) > 0 else np.nan
    return pd.Series(dict(tuiles=len(d), ARAND=d.ARAND.mean(), bord_F=d.bF.mean(),
                          petites=d.s_ok.sum() / max(d.s_n.sum(), 1), p40_200=d.t_ok.sum() / max(d.t_n.sum(), 1),
                          grandes=d.b_ok.sum() / max(d.b_n.sum(), 1), surface=d.surf_ok.sum() / max(d.surf_n.sum(), 1), F1_failles=f1))


if __name__ == "__main__":
    cdir, split, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    files = sorted(glob.glob(f"{cdir}/{split}__*.npz"))
    with Pool(int(os.environ.get("NPROC", 4))) as p:
        rows = [r for rr in p.imap_unordered(run_one, [(f, names) for f in files], chunksize=2) for r in rr]
    D = pd.DataFrame(rows); D.to_pickle(f"seg_eval_{split}_{'_'.join(names)[:80]}.pkl")
    D["groupe"] = np.where(D.kind == "real", "réel GMNA", "synthétique")
    pd.set_option("display.width", 200)
    T = D.groupby(["groupe", "variante"]).apply(summarize).round(4)
    print(T.to_string())
    print("\nTOUT:"); print(D.groupby("variante").apply(summarize).round(4).to_string())
