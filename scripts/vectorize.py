"""GeoRas2Vec – vectorise a georeferenced geological map raster into a GeoPackage (QGIS / ArcGIS).

    python scripts/vectorize.py map.tif legend.csv out.gpkg [--assets DIR] [--lang en|fr] [--tta]

legende.csv : one row per legend entry, columns
    code (required, as printed on the map), nom, age, lithologie (optional),
    and EITHER x0,y0,x1,y1 = pixel box of the legend swatch in carte.tif (best),
    OR couleur_hex = colour of the swatch (e.g. #f2c38a).
Unit names come only from text read on the map and from this legend.
"""
import argparse, os, sys
import numpy as np, pandas as pd, rasterio
from skimage.color import rgb2lab
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import georas2vec
import pipeline as PL, swatch_match as SM, code_reader as CR, legend_reader as LR


def load_legend(csv, rgb):
    L = pd.read_csv(csv, dtype={"code": str}).fillna("")
    feats, hexes, swatches = [], [], []
    for r in L.itertuples():
        if all(hasattr(r, k) and str(getattr(r, k)) != "" for k in ("x0", "y0", "x1", "y1")):
            x0, y0, x1, y1 = (int(float(getattr(r, k))) for k in ("x0", "y0", "x1", "y1"))
            px = rgb[y0:y1, x0:x1].reshape(-1, 3)
            feats.append(PL.swatch_features(rgb2lab(px[None])[0])); hexes.append("#%02x%02x%02x" % tuple(int(v) for v in np.median(px, 0)))
            swatches.append(SM.swatch_image(rgb[y0 + 2:y1 - 2, x0 + 2:x1 - 2]))
        elif getattr(r, "couleur_hex", ""):
            h = r.couleur_hex.lstrip("#"); c = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)]], np.uint8)
            feats.append(PL.swatch_features(np.repeat(rgb2lab(c[None])[0], 50, 0))); hexes.append("#" + h)
            swatches.append(np.tile(c.reshape(1, 1, 3), (SM.PH, SM.PW, 1)))
        else:
            raise SystemExit(f"légende : la ligne {r.code} n'a ni case (x0,y0,x1,y1) ni couleur_hex")
    out = pd.DataFrame(index=L.code.values)
    for c in ("nom", "age", "lithologie"):
        out[c] = L[c].values if c in L else ""
    out["nom"] = np.where(out.nom == "", out.index, out.nom)
    out["couleur_hex"] = hexes; out["feat"] = feats; out["swatch"] = swatches
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raster"); ap.add_argument("legend", help="legend CSV, OR an image of the printed legend (read automatically)")
    ap.add_argument("out")
    ap.add_argument("--legend-box", default=None, help="x0,y0,x1,y1: read the legend automatically from this rectangle of the raster")
    ap.add_argument("--save-legend", default=None, help="write the legend that was read automatically to this CSV (to review / correct)")
    ap.add_argument("--assets", default=None, help="folder with the model files (default: download from Hugging Face)")
    ap.add_argument("--tta", action="store_true", help="average over 4 flips (slower, not better on average)")
    ap.add_argument("--lang", default="en", choices=["en", "fr"], help="language of layer / column names and values (default: en)")
    a = ap.parse_args()
    import easyocr
    A = georas2vec.fetch_assets(a.assets)
    with rasterio.open(a.raster) as s:
        rgb = np.moveaxis(s.read([1, 2, 3]), 0, -1).astype(np.uint8); tr, crs = s.transform, s.crs
    PL.CODE_READER = CR.Reader(A["code_reader"]); PL.SWATCH_ENC = SM.load_encoder(A["swatch_encoder"])
    if a.legend_box or not a.legend.lower().endswith(".csv"):
        if a.legend_box:
            x0, y0, x1, y1 = (int(v) for v in a.legend_box.split(","))
            limg = rgb[y0:y1, x0:x1]
        else:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            limg = np.asarray(Image.open(a.legend).convert("RGB"))
        legend = LR.read_legend_v2(limg, PL.CODE_READER)
        if a.save_legend:
            legend.reset_index()[["code", "nom", "age", "lithologie", "x0", "y0", "x1", "y1", "couleur_hex", "conf"]].to_csv(a.save_legend, index=False)
            print("legend read automatically ->", a.save_legend, "(review it; a corrected CSV can be passed instead)")
    else:
        legend = load_legend(a.legend, rgb)
    print(f"map {rgb.shape[1]}×{rgb.shape[0]} px | legend: {len(legend)} units")
    model = PL.load_model(A["weights"])
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    orig = PL.read_text
    PL.read_text = lambda img, rd, **kw: PL.reread_codes(img, orig(img, rd, **kw))   # codes relus par le lecteur spécialisé
    P, F, lab, pf = PL.run(rgb, tr, crs, model, legend, reader, A["naming"], a.out, tta=a.tta,
                           ranker_pkl=A["ranker"], water_pkl=A["water"], prop_pkl=A["propagation"], lang=a.lang)
    n = P.code.ne("").sum()
    print(f"\n{len(P)} units ({n} named, {len(P) - n} not identified, with suggestions), {len(F)} faults -> {a.out}")
