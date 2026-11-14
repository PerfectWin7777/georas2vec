# Banc des failles vectorisées : carte de probabilité (modèle) -> lignes ; F1 à 2 px contre les failles officielles.
import sys, os, glob, pickle, numpy as np, pandas as pd
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from rasterio.features import rasterize
from rasterio.transform import Affine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL

ZONES = open("e2e_zones.txt").read().split() + ["north_0_2048.pkl"]
os.makedirs("fault_cache", exist_ok=True)


def cache():
    model = None
    for z in ZONES:
        fn = f"fault_cache/{z[:-4]}.npz"
        if os.path.exists(fn):
            continue
        model = model or PL.load_model(os.environ.get("GEORAS2VEC_WEIGHTS", "best_fp16.pt"))
        R = pickle.load(open(f"naming_cache/{z}", "rb"))
        _, pf = PL.infer(model, R["rgb"])
        np.savez_compressed(fn, pf=(pf * 255).astype(np.uint8), gt=R["flt"] > 0)


def trace_graph(sk, min_px):
    """Skeleton -> polylignes entre nœuds (extrémités / jonctions) ; garde les morceaux >= min_px."""
    H, W = sk.shape
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    node = sk & (nb != 2)
    seen_edge = set(); lines = []
    ys, xs = np.nonzero(sk); S = set(zip(ys.tolist(), xs.tolist()))
    def nbrs(p):
        return [(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (p[0] + a, p[1] + b) in S]
    starts = list(zip(*np.nonzero(node))) or []
    visited = np.zeros_like(sk)
    for st in starts:
        for n1 in nbrs(st):
            if (st, n1) in seen_edge:
                continue
            path = [st, n1]; seen_edge.add((st, n1)); seen_edge.add((n1, st)); prev, cur = st, n1
            while not node[cur]:
                nx = [q for q in nbrs(cur) if q != prev and (cur, q) not in seen_edge]
                if not nx:
                    break
                q = nx[0]; seen_edge.add((cur, q)); seen_edge.add((q, cur)); prev, cur = cur, q; path.append(cur)
            for p in path:
                visited[p] = True
            if len(path) >= min_px:
                lines.append(path)
    # boucles fermées sans nœud
    rest = sk & ~visited
    lab, n = ndi.label(rest, np.ones((3, 3)))
    for c in range(1, n + 1):
        pts = list(zip(*np.nonzero(lab == c)))
        if len(pts) >= min_px:
            lines.append(pts)
    return lines


def method(pf, thr, min_px, close=0, graph=True):
    m = pf > thr
    if close:
        m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=close) | m
    sk = skeletonize(m)
    if graph:
        return trace_graph(sk, min_px)
    F = PL.vectorize_faults(pf, Affine.identity(), None, thr=thr, min_px=min_px)
    return [[(int(y), int(x)) for x, y in np.array(g.coords)] for g in F.geometry]


def f1(lines, gt):
    pr = np.zeros(gt.shape, bool)
    for l in lines:
        a = np.array(l)
        for (y0, x0), (y1, x1) in zip(a[:-1], a[1:]):
            n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
            pr[np.linspace(y0, y1, n).round().astype(int).clip(0, gt.shape[0] - 1), np.linspace(x0, x1, n).round().astype(int).clip(0, gt.shape[1] - 1)] = True
    gd, pd_ = ndi.binary_dilation(gt, iterations=2), ndi.binary_dilation(pr, iterations=2)
    return (pr & gd).sum(), (pr & ~gd).sum(), (gt & ~pd_).sum()


if __name__ == "__main__":
    cache()
    VAR = {"actuel (0.2, 15px)": dict(thr=0.2, min_px=15, graph=False),
           "pixels seuls (plafond)": None,
           "graphe 0.2, 8px": dict(thr=0.2, min_px=8), "graphe 0.2, 15px": dict(thr=0.2, min_px=15),
           "graphe 0.2 + fermeture, 8px": dict(thr=0.2, min_px=8, close=1), "graphe 0.15 + fermeture, 8px": dict(thr=0.15, min_px=8, close=1),
           "graphe 0.3 + fermeture, 8px": dict(thr=0.3, min_px=8, close=1), "graphe 0.2 + fermeture x2, 10px": dict(thr=0.2, min_px=10, close=2)}
    for nm, kw in VAR.items():
        tp = fp = fn = 0
        for z in ZONES:
            d = np.load(f"fault_cache/{z[:-4]}.npz"); pf = d["pf"] / 255; gt = d["gt"]
            if kw is None:
                pr = pf > 0.2; gd, pd_ = ndi.binary_dilation(gt, iterations=2), ndi.binary_dilation(pr, iterations=2)
                a, b, c = (pr & gd).sum(), (pr & ~gd).sum(), (gt & ~pd_).sum()
            else:
                a, b, c = f1(method(pf, **kw), gt)
            tp += a; fp += b; fn += c
        p = tp / max(tp + fp, 1); r = 1 - fn / max(fn + tp, 1)
        print(f"{nm:<34} précision {p:6.1%} rappel {r:6.1%} F1 {2*p*r/max(p+r,1e-9):.3f}", flush=True)
