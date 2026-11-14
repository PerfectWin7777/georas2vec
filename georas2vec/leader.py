# Détection des traits de rappel : trait sombre fin qui part du bord d'une étiquette, dans n'importe quelle direction.
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries


def prepare(reg):
    """Masque des pixels 'trait candidat' : sombres, hors contacts, hors failles. Calculé une fois par zone."""
    if "lead_mask" in reg:
        return
    lum, inst = reg["lum"], reg["inst"]
    bd = find_boundaries(inst, mode="thick")
    nearbd = ndi.binary_dilation(bd, iterations=1)
    flt = reg.get("flt")
    nf = ndi.binary_dilation(flt > 0, iterations=3) if flt is not None else np.zeros_like(bd)
    rgb = reg["rgb"].astype(np.int16)
    sat = rgb.max(-1) - rgb.min(-1)
    dark = (lum < 100) & (sat < 70)            # trait noir/gris (pas de texte ou de contour coloré)
    reg["lead_dark"] = dark
    reg["lead_mask"] = dark & ~nearbd & ~nf


def trace(reg, wd, max_len=160, gap=6):
    """Renvoie la liste des extrémités (y, x, longueur) des traits partant de la boîte du mot."""
    prepare(reg)
    M, D, inst = reg["lead_mask"], reg["lead_dark"], reg["inst"]
    N = inst.shape[0]
    x0, y0, x1, y1 = int(max(wd.x0, 0)), int(max(wd.y0, 0)), int(min(wd.x1, N - 1)), int(min(wd.y1, N - 1))
    P = 14
    Y0, X0, Y1, X1 = max(y0 - P, 0), max(x0 - P, 0), min(y1 + P, N - 1), min(x1 + P, N - 1)
    sub = M[Y0:Y1 + 1, X0:X1 + 1].copy()
    # on enlève la boîte de texte (+1 px) : l'encre des lettres n'est pas un trait
    sub[max(y0 - 1 - Y0, 0):y1 + 2 - Y0, max(x0 - 1 - X0, 0):x1 + 2 - X0] = False
    lab, n = ndi.label(sub, structure=np.ones((3, 3)))
    if n == 0:
        return []
    yy, xx = np.nonzero(lab)
    ll = lab[yy, xx]
    gy, gx = yy + Y0, xx + X0
    # distance à la boîte
    db = np.hypot(np.maximum(np.maximum(y0 - gy, gy - y1), 0), np.maximum(np.maximum(x0 - gx, gx - x1), 0))
    out = []
    for c in range(1, n + 1):
        s = ll == c
        if s.sum() < 5 or db[s].min() > 2.5:          # doit toucher (presque) la boîte
            continue
        py, px, d = gy[s].astype(float), gx[s].astype(float), db[s]
        # un trait est fin et allongé
        cov = np.cov(np.stack([py, px]))
        ev, evec = np.linalg.eigh(cov)
        v = evec[:, 1]
        if ev[1] < 9:                               # trop court (< ~10 px)
            continue
        res = np.abs((py - py.mean()) * v[1] - (px - px.mean()) * v[0])
        if np.percentile(res, 90) > 1.3:            # pas droit ou trop épais
            continue
        a = np.argmin(d); b = np.argmax(d)
        if d[b] - d[a] < 4:                           # doit s'éloigner de la boîte
            continue
        if np.dot(v, [py[b] - py[a], px[b] - px[a]]) < 0:
            v = -v
        # suivi dans la direction, au-delà du morceau visible, en tolérant les coupures aux contacts
        y, x = py[b], px[b]; miss = 0; L = float(np.hypot(py[b] - py[a], px[b] - px[a]))
        ey, ex = y, x
        while L < max_len:
            y += v[0]; x += v[1]; L += 1
            iy, ix = int(round(y)), int(round(x))
            if not (0 <= iy < N and 0 <= ix < N):
                break
            if D[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].any():
                miss = 0; ey, ex = y, x
            else:
                miss += 1
                if miss > gap:
                    break
        # pointe : on avance de 2 px au-delà de la fin (le trait se termine DANS le polygone visé)
        ty, tx = int(np.clip(round(ey + 2 * v[0]), 0, N - 1)), int(np.clip(round(ex + 2 * v[1]), 0, N - 1))
        out.append((ty, tx, float(np.hypot(ty - py[a], tx - px[a]))))
    return out
