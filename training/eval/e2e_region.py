# Bout en bout sur une GRANDE région (plusieurs zones de 2048 px assemblées) : la propagation voit toute la région.
#   python e2e_big.py <r0> <c0> <n_zones_cote> <sortie_dir>
import sys, os, glob, pickle, re, time
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
import numpy as np, pandas as pd, rasterio, easyocr
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL
r0, c0, n, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(OUT, exist_ok=True)
W = os.environ.get("GEORAS2VEC_RAW", "data/world_dataset/")
norm = lambda c: re.sub(r"[\*\?\s]", "", str(c))
S = 2048 * n
rgb = np.zeros((S, S, 3), np.uint8); tcode = np.full((S, S), -1, np.int32); codes = {}
for i in range(n):
    for j in range(n):
        R = pickle.load(open(f"naming_cache/north_{r0 + 2048 * i}_{c0 + 2048 * j}.pkl", "rb"))
        rgb[2048 * i:2048 * (i + 1), 2048 * j:2048 * (j + 1)] = R["rgb"]
        tr_ = np.array([-1] + [codes.setdefault(norm(c), len(codes)) for c in R["codes"]])
        tcode[2048 * i:2048 * (i + 1), 2048 * j:2048 * (j + 1)] = tr_[R["inst"]]
inv = np.array(list(codes))
with rasterio.open(W + "na_north/gmna_north_300dpi.tif") as s:
    tr = rasterio.windows.transform(Window(c0, r0, S, S), s.transform); crs = s.crs
LEG = pd.read_pickle("legend_gmna.pkl")
z = pickle.load(open(f"bench_cache/north_{r0}_{c0}.pkl", "rb"))
local = LEG.reindex(z["legend"].index).copy(); local["feat"] = z["legend"].feat.values
for c in ("nom", "age", "lithologie", "couleur_hex"):
    local[c] = local[c].fillna(local.index.to_series() if c == "nom" else "")
model = PL.load_model(os.environ.get("GEORAS2VEC_WEIGHTS", "best_fp16.pt"))
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
t = time.time()
P, F, lab, pf = PL.run(rgb, tr, crs, model, local, reader, "naming_model.pkl", f"{OUT}/carte_geologique.gpkg",
                       ranker_pkl="legend_ranker_sud.pkl", water_pkl="water_model_sud.pkl", prop_pkl=os.environ.get("PROP", "prop_model_pred.pkl"))
print("durée totale %.0f s" % (time.time() - t))
valid = tcode >= 0
pickle.dump(dict(lab0=PL.LAST["lab0"], N0=PL.LAST["N0"], N_prop=PL.LAST["N_prop"], lab1=PL.LAST["lab1"], N1=PL.LAST["N1"]), open(f"{OUT}/etat.pkl", "wb"))


def evaluate(lab_, N, nm):
    code = np.asarray(N.code.reindex(np.arange(lab_.max() + 1)).fillna("").astype(object).values, dtype=object)
    srcs = np.asarray(N.source_nom.reindex(np.arange(lab_.max() + 1)).fillna("non identifiée").astype(object).values, dtype=object)
    cid = np.array([codes.get(c, -2) if c else -3 for c in code])
    pc = cid[lab_]; named = valid & (pc != -3); ok = named & (pc == tcode)
    print(f"\n[{nm}] polygones {len(N)} | surface nommée {named.sum()/valid.sum():.1%} | juste/nommée {ok.sum()/max(named.sum(),1):.1%} | juste/total {ok.sum()/valid.sum():.1%}")
    sp = srcs[lab_]
    for s_ in sorted(set(sp[valid])):
        m = valid & (sp == s_)
        print(f"   {s_:<45} {m.sum()/valid.sum():6.1%} de la surface, juste {(ok & m).sum()/max(m.sum(),1):6.1%}")
    return dict(nommee=named.sum() / valid.sum(), juste=ok.sum() / max(named.sum(), 1), juste_total=ok.sum() / valid.sum(), n=len(N))


a = evaluate(PL.LAST["lab0"], PL.LAST["N0"], "SANS propagation")
b = evaluate(PL.LAST["lab0"], PL.LAST["N_prop"], "AVEC propagation (avant fusion)")
c = evaluate(PL.LAST["lab1"], PL.LAST["N1"], "AVEC propagation + fusion des voisins de même code")
pd.DataFrame([a, b, c], index=["sans", "propagation", "propagation+fusion"]).to_csv(f"{OUT}/resume.csv")
pickle.dump(dict(lab0=PL.LAST["lab0"], N0=PL.LAST["N0"], N_prop=PL.LAST["N_prop"]), open(f"{OUT}/etat.pkl", "wb"))
