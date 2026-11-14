import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

def segments(pf, thr=0.15, close=1, min_px=6):
    """Squelette -> segments entre jonctions, ordonnés, raccrochés à leurs jonctions. Retourne des listes de (y, x)."""
    m = pf > thr
    if close:
        m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=close) | m
    sk = skeletonize(m)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    junc = sk & (nb >= 3)
    jd = ndi.binary_dilation(junc, np.ones((3, 3)))
    body = sk & ~jd
    lab, n = ndi.label(body, np.ones((3, 3)))
    jl, _ = ndi.label(jd & sk, np.ones((3, 3)))
    jc = ndi.center_of_mass(np.ones_like(jl), jl, np.arange(1, jl.max() + 1)) if jl.max() else []
    objs = ndi.find_objects(lab)
    out = []
    for c, sl in enumerate(objs, 1):
        if sl is None:
            continue
        sub = lab[sl] == c; ys, xs = np.nonzero(sub); ys = ys + sl[0].start; xs = xs + sl[1].start
        if len(ys) < min_px:
            continue
        P = set(zip(ys.tolist(), xs.tolist()))
        deg = {p: sum((p[0] + a, p[1] + b) in P for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b) for p in P}
        st = min(P, key=lambda p: (deg[p], p))
        path, seen, cur = [st], {st}, st
        while True:
            nx = [(cur[0] + a, cur[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (cur[0] + a, cur[1] + b) in P and (cur[0] + a, cur[1] + b) not in seen]
            if not nx:
                break
            nx.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1])); cur = nx[0]; seen.add(cur); path.append(cur)
        # raccrocher aux jonctions voisines (à moins de 3 px des extrémités)
        for end, ins in ((path[0], 0), (path[-1], None)):
            y0, x0 = end
            win = jl[max(0, y0 - 3):y0 + 4, max(0, x0 - 3):x0 + 4]
            ids = np.unique(win[win > 0])
            if len(ids):
                cy, cx = jc[ids[0] - 1]; pt = (int(round(cy)), int(round(cx)))
                if ins == 0: path.insert(0, pt)
                else: path.append(pt)
        out.append(path)
    return out
