# Banc de test du NOMMAGE seul (polygones de référence, OCR en cache).
# Étape 1 (ce script) : pour chaque zone, on prépare : signature de chaque polygone, vrai code, surface,
#   légende "de la feuille" (cases = signatures médianes des autres zones de la même feuille, zone testée exclue),
#   candidats OCR notés par le modèle appris.  -> bench_cache/<zone>.pkl
import sys, os, glob, pickle, re, numpy as np, pandas as pd
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
from skimage.color import rgb2lab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL
norm = lambda c: re.sub(r"[\*\?\s]", "", str(c))
BD = os.environ.get("BENCH_DIR", "bench_cache"); os.makedirs(BD, exist_ok=True)
files = sorted(glob.glob("naming_cache/north_*.pkl") + glob.glob("naming_cache/south_*.pkl"))
sheet = lambda f: os.path.basename(f).split("_")[0]

per, polys = {}, {}
for f in files:
    R = pickle.load(open(f, "rb")); inst = R["inst"]; L = rgb2lab(R["rgb"]).reshape(-1, 3); area = np.bincount(inst.ravel())
    order = np.argsort(inst.ravel(), kind="stable"); sl = inst.ravel()[order]; d = {}; rows = []
    for i in np.nonzero(area)[0]:
        if i == 0:
            continue
        a, b = np.searchsorted(sl, i), np.searchsorted(sl, i, side="right")
        ft = PL.swatch_features(L[order[a:b]]); c = norm(R["codes"][i - 1])
        rows.append(dict(poly=i, code=c, area=int(area[i]), feat=ft))
        if area[i] >= 400:
            d.setdefault(c, []).append(ft)
    per[f] = d; polys[f] = pd.DataFrame(rows)
    print("signatures", f, len(rows), flush=True)

for f in files:
    out = f"{BD}/{os.path.basename(f)}"
    if os.path.exists(out):
        continue
    R = pickle.load(open(f, "rb"))
    leg = {}
    for g, d in per.items():
        if g == f or sheet(g) != sheet(f):
            continue
        for c, v in d.items():
            leg.setdefault(c, []).extend(v)
    legend = pd.DataFrame({c: dict(feat=np.median(np.array(v), 0)) for c, v in leg.items()}).T
    D, ch = PL.ocr_candidates(R["rgb"], R["inst"], R["words"], legend.index, "naming_model.pkl", R.get("flt"))
    pickle.dump(dict(polys=polys[f], legend=legend, D=D, ch=ch, present=set(polys[f].code)), open(out, "wb"))
    print("ocr", f, len(D), flush=True)
