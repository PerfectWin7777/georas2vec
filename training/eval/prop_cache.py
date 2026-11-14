# Piste 1 (propagation par la couleur de la carte) — étape 1 : pour chaque zone NORD (jamais vue par le nommage),
# nommage actuel sur les polygones de référence + signature couleur/texture de chaque polygone + vrai code.
import sys, os, glob, pickle, re, numpy as np, pandas as pd
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
from scipy import ndimage as ndi
from skimage.color import rgb2lab
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import pipeline as PL
norm = lambda c: re.sub(r"[\*\?\s]", "", str(c))
os.makedirs("prop_cache", exist_ok=True)
for f in sorted(glob.glob("naming_cache/north_*.pkl")):
    name = os.path.basename(f)[:-4]; fn = f"prop_cache/{name}.pkl"
    if os.path.exists(fn) or not os.path.exists(f"bench_cache/{name}.pkl"):
        continue
    R = pickle.load(open(f, "rb")); inst = R["inst"]; z = pickle.load(open(f"bench_cache/{name}.pkl", "rb"))
    leg = z["legend"]
    N = PL.name_polygons(R["rgb"], inst, R["words"], leg, "naming_model.pkl", flt=(R["flt"] > 0).astype(np.uint8), ranker_pkl="legend_ranker_sud.pkl")
    labc = rgb2lab(R["rgb"]).reshape(-1, 3)
    ids = np.unique(inst[inst > 0]); order = np.argsort(inst.ravel(), kind="stable"); sl = inst.ravel()[order]
    st, en = np.searchsorted(sl, ids), np.searchsorted(sl, ids, side="right")
    F = np.stack([PL.swatch_features(labc[order[a:b]]) for a, b in zip(st, en)])
    cy, cx = np.array(ndi.center_of_mass(np.ones_like(inst), inst, ids)).T
    P = pd.DataFrame(dict(poly=ids, aire=en - st, cy=cy, cx=cx, vrai=[norm(R["codes"][i - 1]) for i in ids]))
    P = P.join(N, on="poly")
    pickle.dump(dict(P=P, F=F), open(fn, "wb")); print(name, len(P), flush=True)
