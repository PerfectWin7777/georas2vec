import os as _o, sys as _s
_R = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..") if "eval" in __file__ else _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..")
_s.path[:0] = [_o.path.join(_R, "georas2vec"), _o.path.join(_R, "training", "eval"), _o.path.join(_R, "training")]
# Test du nommage v0.8 sur des zones réelles (vérité connue) :
#   (a) légende actuelle (couleurs mesurées sur l'autre feuille), OCR générique
#   (b) légende LUE AUTOMATIQUEMENT (notice USGS : cases + codes), OCR générique
#   (c) (b) + lecteur de codes géologiques pour la légende ET pour les textes de la carte
#   python naming_v08_test.py north_0_14336 [autres zones...]
import sys, os, pickle, re, numpy as np, pandas as pd, rasterio, easyocr
from rasterio.windows import Window
from skimage.color import rgb2lab
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "."); import pipeline as PL, code_reader as CR, legend_reader as LR
W_ = os.environ.get("GEORAS2VEC_RAW", "data/world_dataset/")
canon = lambda c: CR.canon(re.sub(r"[\?\s]", "", str(c))).rstrip("*")


def auto_legend(reader_kind):
    fn = f"legend_auto_{reader_kind}.pkl"
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    rgbL = np.asarray(Image.open("../dl_ds424/legend200-1.png").convert("RGB"))
    S = pd.read_pickle("../dl_ds424/swatches.pkl").reset_index(drop=True)
    if reader_kind == "crnn":
        R = CR.Reader("code_reader.pt")
        out = R.read([Image.fromarray(rgbL[int(r.y0) + 2:int(r.y1) - 2, int(r.x0) + 2:int(r.x1) - 2]) for r in S.itertuples()])
        S["code"] = [o[0] for o in out]; S["conf"] = [o[1] for o in out]
    else:
        S = pd.read_pickle("../dl_ds424/swatches_read.pkl"); S["code"] = S.code.map(lambda c: CR.canon(c))
    feats, hexes = [], []
    for r in S.itertuples():
        px = rgbL[int(r.y0) + 3:int(r.y1) - 3, int(r.x0) + 3:int(r.x1) - 3].reshape(-1, 3); px = px[px.max(1) >= 110]
        feats.append(PL.swatch_features(rgb2lab(px[None].astype(np.uint8))[0])); hexes.append("#%02x%02x%02x" % tuple(int(v) for v in np.median(px, 0)))
    S["feat"] = feats; S["couleur_hex"] = hexes
    S = S[S.code.str.len() > 0].sort_values("conf", ascending=False).drop_duplicates("code")
    L = pd.DataFrame(dict(nom=S.code.values, age="", lithologie="", couleur_hex=S.couleur_hex.values, feat=list(S.feat.values)), index=S.code.values)
    L.to_pickle(fn); return L


def cross_sheet_legend(zone):
    LEG = pd.read_pickle("legend_gmna.pkl"); z = pickle.load(open(f"bench_cache/{zone}.pkl", "rb"))
    L = LEG.reindex(z["legend"].index).copy(); L["feat"] = z["legend"].feat.values
    for c in ("nom", "age", "lithologie", "couleur_hex"):
        L[c] = L[c].fillna(L.index.to_series() if c == "nom" else "")
    return L


def evaluate(zone, lab, N, R):
    true = np.array([""] + [canon(c) for c in R["codes"]])[R["inst"]]
    code = np.full(lab.max() + 1, "", dtype=object); code[N.index] = [canon(c) for c in N.code]
    pc = code[lab]; valid = R["inst"] > 0; named = valid & (pc != ""); ok = named & (pc == true)
    return named.sum() / valid.sum(), ok.sum() / max(named.sum(), 1), ok.sum() / valid.sum()


if __name__ == "__main__":
    model = PL.load_model(os.environ.get("GEORAS2VEC_WEIGHTS", "best_fp16.pt")); reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    res = []
    for zone in sys.argv[1:]:
        R = pickle.load(open(f"naming_cache/{zone}.pkl", "rb")); part, r0, c0 = zone.split("_")
        with rasterio.open(W_ + f"na_{part}/gmna_{part}_300dpi.tif") as s:
            tr = rasterio.windows.transform(Window(int(c0), int(r0), 2048, 2048), s.transform); crs = s.crs
        for nm, leg, use_cr in (("(a) légende actuelle", cross_sheet_legend(zone), False),
                                ("(b) légende lue (OCR générique)", auto_legend("easyocr"), False),
                                ("(c) légende + carte lues par notre lecteur", auto_legend("crnn"), True)):
            PL.CODE_READER = CR.Reader("code_reader.pt") if use_cr else None
            orig = PL.read_text
            if use_cr:
                PL.read_text = lambda rgb, rd, **kw: PL.reread_codes(rgb, orig(rgb, rd, **kw))
            PL.run(R["rgb"], tr, crs, model, leg, reader, "naming_model.pkl", f"../vec_test/{zone}_{nm[1]}.gpkg",
                   ranker_pkl="legend_ranker_sud.pkl", water_pkl="water_model_sud.pkl", prop_pkl="prop_model.pkl", log=lambda *a: None)
            PL.read_text = orig
            a = evaluate(zone, PL.LAST["lab1"], PL.LAST["N1"], R)
            res.append(dict(zone=zone, config=nm, nommee=a[0], juste=a[1], juste_total=a[2]))
            print(f"{zone} {nm:<46} nommée {a[0]:.1%}  juste/nommée {a[1]:.1%}  juste/total {a[2]:.1%}", flush=True)
    D = pd.DataFrame(res); D.to_csv("naming_v08_test.csv", index=False)
    print(D.groupby("config")[["nommee", "juste", "juste_total"]].mean().round(3).to_string())
