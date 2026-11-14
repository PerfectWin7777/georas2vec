"""Lecture automatique d'une légende de carte géologique : cases de couleur (rectangles bordés) + code imprimé dedans ou à côté.

    python legend_reader.py legende.png sortie.csv [--debug debug.png]

Sortie : CSV au format de vectorize.py (code, nom, age, lithologie, x0, y0, x1, y1, couleur_hex), une ligne par case.
Un même code peut avoir plusieurs cases (ex. 'TQ' à terre et 'TQ*' en mer).
"""
import re, sys, argparse
import numpy as np, pandas as pd
from scipy import ndimage as ndi


def find_swatches(rgb, min_w=30, max_w=400, min_h=18, max_h=160):
    """Rectangles de légende : composantes connexes de pixels NON noirs (l'intérieur d'une case, trouée par le texte)
    dont la boîte englobante est bien remplie et entourée d'un trait sombre."""
    dark = rgb.max(2) < 110                                    # traits et texte noirs (les motifs colorés ne le sont pas)
    lab, n = ndi.label(~dark, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    sl = ndi.find_objects(lab)
    area = np.bincount(lab.ravel())
    out = []
    for i, s in enumerate(sl, 1):
        if s is None:
            continue
        h, w = s[0].stop - s[0].start, s[1].stop - s[1].start
        if not (min_w <= w <= max_w and min_h <= h <= max_h) or not (1.2 <= w / h <= 5.0):
            continue
        fill = area[i] / (w * h)
        if fill < 0.55:
            continue
        # bordure : le cadre juste autour de la boîte doit être majoritairement sombre
        y0, y1, x0, x1 = s[0].start, s[0].stop, s[1].start, s[1].stop
        ring = np.concatenate([dark[max(0, y0 - 1), x0:x1], dark[min(y1, dark.shape[0] - 1), x0:x1],
                               dark[y0:y1, max(0, x0 - 1)], dark[y0:y1, min(x1, dark.shape[1] - 1)]])
        if ring.size == 0 or ring.mean() < 0.6:
            continue
        out.append((x0, y0, x1, y1, fill))
    return pd.DataFrame(out, columns=["x0", "y0", "x1", "y1", "fill"])


def clean_code(t):
    t = t.strip().replace(" ", "").replace("|", "l")
    return re.sub(r"[^\w\*²³¹⁰-⁹ЄƐʀƦɆ]", "", t)


def read_codes(rgb, S, reader, mag=3, log=None):
    """Code imprimé DANS chaque case : détection + lecture sur la case agrandie en couleur (meilleur réglage mesuré :
    28/60 exacts, 39/60 proches sur la légende USGS ; les symboles Є, Ʀ, exposants restent difficiles pour un OCR générique)."""
    from skimage.transform import resize
    codes, confs = [], []
    for k, r in enumerate(S.itertuples()):
        crop = rgb[r.y0 + 3:r.y1 - 3, r.x0 + 3:r.x1 - 3]
        big = resize(crop, (crop.shape[0] * mag, crop.shape[1] * mag, 3), order=1, preserve_range=True).astype(np.uint8)
        res = [x for x in reader.readtext(big, detail=1) if x[2] > 0.1]
        if not res:
            codes.append(""); confs.append(0.0)
        else:
            best = max(res, key=lambda x: (x[0][2][0] - x[0][0][0]) * x[2])
            codes.append(clean_code(best[1])); confs.append(float(best[2]))
        if log and k % 100 == 0:
            log(f"{k}/{len(S)}")
    S = S.copy(); S["code"] = codes; S["conf"] = confs
    return S


def swatch_colours(rgb, S):
    from skimage.color import rgb2lab
    hexes, feats = [], []
    for r in S.itertuples():
        px = rgb[r.y0 + 3:r.y1 - 3, r.x0 + 3:r.x1 - 3].reshape(-1, 3)
        px = px[px.max(1) >= 110]                              # sans le texte noir
        if len(px) < 10:
            px = rgb[r.y0:r.y1, r.x0:r.x1].reshape(-1, 3)
        hexes.append("#%02x%02x%02x" % tuple(int(v) for v in np.median(px, 0)))
    S = S.copy(); S["couleur_hex"] = hexes
    return S


def read_legend_v2(rgb, code_reader):
    """Légende complète : cases détectées, codes lus par le lecteur de codes géologiques, couleur, signature, image de case.
    Retourne un DataFrame au format de pipeline.run (index = code)."""
    from PIL import Image
    from skimage.color import rgb2lab
    import pipeline as PL, swatch_match as SM
    S = find_swatches(rgb)
    crops = [rgb[int(r.y0) + 2:int(r.y1) - 2, int(r.x0) + 2:int(r.x1) - 2] for r in S.itertuples()]
    reads = code_reader.read([Image.fromarray(c) for c in crops])
    rows = []
    for r, c, (code, p) in zip(S.itertuples(), crops, reads):
        if not code:
            continue
        px = c.reshape(-1, 3); px = px[px.max(1) >= 110]
        if len(px) < 10:
            px = c.reshape(-1, 3)
        rows.append(dict(code=code, conf=p, x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1,
                         couleur_hex="#%02x%02x%02x" % tuple(int(v) for v in np.median(px, 0)),
                         feat=PL.swatch_features(rgb2lab(px[None].astype(np.uint8))[0]), swatch=SM.swatch_image(c)))
    D = pd.DataFrame(rows).sort_values("conf", ascending=False).drop_duplicates("code").set_index("code")
    D["nom"] = D.index; D["age"] = ""; D["lithologie"] = ""
    return D


def read_legend(rgb, reader=None):
    import easyocr
    reader = reader or easyocr.Reader(["en"], gpu=False, verbose=False)
    S = find_swatches(rgb)
    S = read_codes(rgb, S, reader)
    S = swatch_colours(rgb, S)
    S = S[S.code != ""].reset_index(drop=True)
    S["nom"] = S.code; S["age"] = ""; S["lithologie"] = ""
    return S[["code", "nom", "age", "lithologie", "x0", "y0", "x1", "y1", "couleur_hex", "conf"]]


if __name__ == "__main__":
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("out"); a = ap.parse_args()
    rgb = np.asarray(Image.open(a.image).convert("RGB"))
    L = read_legend(rgb)
    L.to_csv(a.out, index=False); print(len(L), "cases ->", a.out)
