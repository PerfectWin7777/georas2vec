"""Fond topographique synthétique, dessiné PAR-DESSUS une tuile de carte géologique (augmentation d'entraînement).

Les cartes géologiques détaillées (ex. BRGM 1/50 000) impriment la géologie sur un fond topographique : courbes de niveau,
routes, chemins, haies et limites de parcelles, bâtiments, toponymes, points cotés, quadrillage. Ces traits ne sont
NI des limites d'unités NI des failles : les étiquettes de la tuile ne changent pas, le réseau apprend à les ignorer.
"""
import random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
SYL = ["la", "le", "mon", "bois", "ville", "ham", "eau", "ferme", "fos", "ter", "nie", "rie", "mes", "nil", "court", "val",
       "bray", "pont", "mou", "lin", "sain", "vin", "char", "les", "bar", "on", "ière", "ot", "aux", "mont"]


def _name(rng):
    w = "".join(rng.choice(SYL) for _ in range(rng.randint(2, 4)))
    return (rng.choice(["la ", "le ", "les ", "", "", "St-"]) + w).title() if rng.random() < 0.8 else w.upper()


def _elevation(h, w, rng):
    z = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 1, (h // 8 + 2, w // 8 + 2))
    z = ndi.gaussian_filter(z, rng.uniform(2, 6))
    z = np.array(Image.fromarray(((z - z.min()) / (np.ptp(z) + 1e-9) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32)
    return z


def _polyline(rng, w, h, n, step, wiggle):
    x, y = rng.uniform(0, w), rng.uniform(0, h); a = rng.uniform(0, 2 * math.pi); pts = [(x, y)]
    for _ in range(n):
        a += rng.gauss(0, wiggle); x += step * math.cos(a); y += step * math.sin(a); pts.append((x, y))
    return pts


def topo_overlay(rgb, rng=None, strength=None):
    """rgb : uint8 HxWx3 -> même tuile avec un fond topographique dessiné par-dessus."""
    rng = rng or random.Random()
    h, w = rgb.shape[:2]; s = strength if strength is not None else rng.uniform(0.5, 1.0)
    ink = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(ink)
    brown = (rng.randint(120, 170), rng.randint(70, 100), rng.randint(30, 60))
    # courbes de niveau (brunes), courbes maîtresses plus épaisses
    if rng.random() < 0.85:
        z = _elevation(h, w, rng); eq = rng.choice([3, 4, 5, 6, 8, 10, 12])
        lev = np.floor(z / eq); edge = np.zeros((h, w), bool)
        edge[:, 1:] |= lev[:, 1:] != lev[:, :-1]; edge[1:, :] |= lev[1:, :] != lev[:-1, :]
        major = np.zeros((h, w), bool); lm = np.floor(z / (eq * 5)); major[:, 1:] |= lm[:, 1:] != lm[:, :-1]; major[1:, :] |= lm[1:, :] != lm[:-1, :]
        a = np.zeros((h, w, 4), np.uint8); a[edge] = brown + (int(150 * s),); a[ndi.binary_dilation(major)] = brown + (int(210 * s),)
        ink = Image.alpha_composite(ink, Image.fromarray(a)); d = ImageDraw.Draw(ink)
    # limites de parcelles / haies : réseau de segments fins, noirs ou bruns, souvent pointillés
    for _ in range(int(rng.uniform(0, 160 if rng.random() < 0.4 else 60) * s)):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h); L = rng.uniform(20, 120); a_ = rng.uniform(0, math.pi)
        col = (30, 30, 30, int(200 * s)) if rng.random() < 0.6 else brown + (int(200 * s),)
        x1, y1 = x0 + L * math.cos(a_), y0 + L * math.sin(a_)
        if rng.random() < 0.4:                                                          # haie : pointillés
            for t in np.arange(0, 1, 0.08):
                d.ellipse((x0 + (x1 - x0) * t - 1, y0 + (y1 - y0) * t - 1, x0 + (x1 - x0) * t + 1, y0 + (y1 - y0) * t + 1), fill=col)
        else:
            d.line((x0, y0, x1, y1), fill=col, width=1)
    # routes (doubles traits, parfois remplies de rouge/jaune), chemins (tirets)
    for _ in range(rng.randint(0, 6)):
        pts = _polyline(rng, w, h, rng.randint(8, 30), rng.uniform(15, 40), 0.25)
        wd = rng.choice([1, 2, 3, 4])
        if wd >= 3:
            d.line(pts, fill=(25, 25, 25, int(230 * s)), width=wd + 2)
            d.line(pts, fill=rng.choice([(250, 250, 245), (230, 70, 60), (250, 200, 60)]) + (int(255 * s),), width=wd)
        else:
            d.line(pts, fill=(25, 25, 25, int(220 * s)), width=wd)
    for _ in range(rng.randint(0, 8)):
        pts = _polyline(rng, w, h, rng.randint(6, 20), rng.uniform(8, 20), 0.4)
        for p, q in zip(pts[::2], pts[1::2]):
            d.line((p, q), fill=(40, 40, 40, int(200 * s)), width=1)
    # villages : grappes de petits bâtiments noirs
    for _ in range(rng.randint(0, 5)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        for _ in range(rng.randint(5, 40)):
            x, y = cx + rng.gauss(0, 25), cy + rng.gauss(0, 25); bw, bh = rng.uniform(2, 7), rng.uniform(2, 7)
            d.polygon([(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)], fill=(20, 20, 20, int(230 * s)))
    # toponymes, points cotés, numéros de routes
    for _ in range(rng.randint(0, 14)):
        f = ImageFont.truetype(rng.choice(FONTS), rng.randint(11, 28)); x, y = rng.uniform(-20, w), rng.uniform(0, h)
        k = rng.random()
        if k < 0.6:
            txt, col = _name(rng), (25, 25, 25, int(240 * s))
        elif k < 0.85:
            txt, col = str(rng.randint(5, 650)), brown + (int(230 * s),); d.ellipse((x - 2, y + 6, x + 1, y + 9), fill=col)
        else:
            txt, col = rng.choice(["D", "N", "C", "GR "]) + str(rng.randint(1, 999)), (30, 30, 30, int(230 * s))
        d.text((x, y), txt, font=f, fill=col)
    # quadrillage kilométrique / amorces
    if rng.random() < 0.3:
        step = rng.randint(150, 400); c = (40, 40, 160, int(150 * s))
        for g in range(rng.randint(0, step), w, step):
            d.line((g, 0, g, h), fill=c, width=1)
        for g in range(rng.randint(0, step), h, step):
            d.line((0, g, w, g), fill=c, width=1)
    out = Image.alpha_composite(Image.fromarray(rgb).convert("RGBA"), ink).convert("RGB")
    return np.asarray(out)


def restyle_faults(rgb, flt, rng=None):
    """Redessine les failles de la vérité en traits NOIRS FINS (parfois tiretés), style des cartes détaillées françaises :
    le réseau apprend qu'une faille peut ressembler à une route ou un chemin."""
    rng = rng or random.Random()
    m = flt > 0
    if not m.any():
        return rgb
    if rng.random() < 0.5:
        m = ndi.binary_dilation(m, iterations=1)
    if rng.random() < 0.4:                                     # tirets : on coupe le tracé par blocs
        yy, xx = np.nonzero(m); period = rng.randint(6, 14)
        keep = ((yy // period + xx // period) % 2) == 0; m2 = np.zeros_like(m); m2[yy[keep], xx[keep]] = True; m = m2
    out = rgb.copy(); c = rng.randint(0, 50)
    out[m] = (np.array([c, c, c]) * 0.85 + out[m] * 0.15).astype(np.uint8)
    return out
