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
import pipeline as PL


def load_legend(csv, rgb):
    L = pd.read_csv(csv, dtype={"code": str}).fillna("")
    lab = None
    feats, hexes = [], []
    for r in L.itertuples():
        if all(hasattr(r, k) and str(getattr(r, k)) != "" for k in ("x0", "y0", "x1", "y1")):
            x0, y0, x1, y1 = (int(float(getattr(r, k))) for k in ("x0", "y0", "x1", "y1"))
            px = rgb[y0:y1, x0:x1].reshape(-1, 3)
            feats.append(PL.swatch_features(rgb2lab(px[None])[0])); hexes.append("#%02x%02x%02x" % tuple(int(v) for v in np.median(px, 0)))
        elif getattr(r, "couleur_hex", ""):
            h = r.couleur_hex.lstrip("#"); c = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)]], np.uint8)
            feats.append(PL.swatch_features(np.repeat(rgb2lab(c[None])[0], 50, 0))); hexes.append("#" + h)
        else:
            raise SystemExit(f"légende : la ligne {r.code} n'a ni case (x0,y0,x1,y1) ni couleur_hex")
    out = pd.DataFrame(index=L.code.values)
    for c in ("nom", "age", "lithologie"):
        out[c] = L[c].values if c in L else ""
    out["nom"] = np.where(out.nom == "", out.index, out.nom)
    out["couleur_hex"] = hexes; out["feat"] = feats
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raster"); ap.add_argument("legend"); ap.add_argument("out")
    ap.add_argument("--assets", default=None, help="folder with the model files (default: download from Hugging Face)")
    ap.add_argument("--tta", action="store_true", help="average over 4 flips (slower, not better on average)")
    ap.add_argument("--lang", default="en", choices=["en", "fr"], help="language of layer / column names and values (default: en)")
    a = ap.parse_args()
    import easyocr
    A = georas2vec.fetch_assets(a.assets)
    with rasterio.open(a.raster) as s:
        rgb = np.moveaxis(s.read([1, 2, 3]), 0, -1).astype(np.uint8); tr, crs = s.transform, s.crs
    legend = load_legend(a.legend, rgb)
    print(f"map {rgb.shape[1]}×{rgb.shape[0]} px | legend: {len(legend)} units")
    model = PL.load_model(A["weights"])
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    P, F, lab, pf = PL.run(rgb, tr, crs, model, legend, reader, A["naming"], a.out, tta=a.tta,
                           ranker_pkl=A["ranker"], water_pkl=A["water"], prop_pkl=A["propagation"], lang=a.lang)
    n = P.code.ne("").sum()
    print(f"\n{len(P)} units ({n} named, {len(P) - n} not identified, with suggestions), {len(F)} faults -> {a.out}")
