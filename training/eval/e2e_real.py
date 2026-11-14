import os as _o, sys as _s
_R = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..") if "eval" in __file__ else _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..")
_s.path[:0] = [_o.path.join(_R, "georas2vec"), _o.path.join(_R, "training", "eval"), _o.path.join(_R, "training")]
# Bout en bout RÉALISTE (v0.7) sur des zones de la feuille SUD : polygones prédits, vraie légende imprimée lue
# automatiquement (cases + codes), lecteur de codes sur la carte, propagation. Classeurs entraînés sur le NORD seulement,
# comparateur de cases entraîné sur le NORD seulement.
#   python e2e_real.py south_2048_10240 ...
import sys, os, pickle, numpy as np, pandas as pd, rasterio, easyocr
from rasterio.windows import Window
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "."); import pipeline as PL, code_reader as CR, legend_reader as LR, swatch_match as SM
from naming_v08_test import canon, evaluate
W_ = os.environ.get("GEORAS2VEC_RAW", "data/world_dataset/")
CRD = CR.Reader("code_reader.pt")
if os.path.exists("legend_real.pkl"):
    LEG = pd.read_pickle("legend_real.pkl")
else:
    LEG = LR.read_legend_v2(np.asarray(Image.open("../dl_ds424/legend200-1.png").convert("RGB")), CRD); LEG.to_pickle("legend_real.pkl")
print("légende lue automatiquement :", len(LEG), "codes", flush=True)
model = PL.load_model(os.environ.get("GEORAS2VEC_WEIGHTS", "best_fp16.pt")); reader = easyocr.Reader(["en"], gpu=False, verbose=False)
orig = PL.read_text
PL.read_text = lambda rgb, rd, **kw: PL.reread_codes(rgb, orig(rgb, rd, **kw))
PL.CODE_READER = CRD; PL.SWATCH_ENC = SM.load_encoder("swatch_enc.pt")
res = []
for zone in sys.argv[1:]:
    R = pickle.load(open(f"naming_cache/{zone}.pkl", "rb")); part, r0, c0 = zone.split("_")
    with rasterio.open(W_ + f"na_{part}/gmna_{part}_300dpi.tif") as s:
        tr = rasterio.windows.transform(Window(int(c0), int(r0), 2048, 2048), s.transform); crs = s.crs
    for nm, rk in (("vraie légende, sans comparateur", "legend_ranker_real_nord.pkl"), ("vraie légende + comparateur (v0.7)", "legend_ranker_sw_nord.pkl")):
        PL.run(R["rgb"], tr, crs, model, LEG, reader, "naming_model.pkl", f"../vec_test/{zone}_real.gpkg",
               ranker_pkl=rk, water_pkl="", prop_pkl="prop_model.pkl", log=lambda *a: None)
        a0 = evaluate(zone, PL.LAST["lab0"], PL.LAST["N0"], R); a1 = evaluate(zone, PL.LAST["lab1"], PL.LAST["N1"], R)
        res.append(dict(zone=zone, config=nm, nommee=a1[0], juste=a1[1], juste_total=a1[2], juste_total_avant_prop=a0[2]))
        print(f"{zone} {nm:<36} texte+légende {a0[2]:.1%} | + propagation : nommée {a1[0]:.1%} juste {a1[1]:.1%} juste/total {a1[2]:.1%}", flush=True)
D = pd.DataFrame(res); D.to_csv("e2e_real.csv", index=False)
print(D.groupby("config")[["juste_total_avant_prop", "nommee", "juste", "juste_total"]].mean().round(3).to_string())
