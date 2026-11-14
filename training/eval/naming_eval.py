# Évaluation du nommage avec la fonction EXACTE du pipeline (PL.rank_pairs), validation croisée entre feuilles.
#   python naming_eval4.py <bench_dir> <sortie.pkl>
import sys, os, glob, pickle, numpy as np, pandas as pd
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
from scipy import ndimage as ndi
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import pipeline as PL
from sklearn.ensemble import HistGradientBoostingClassifier
BD, OUT = sys.argv[1], sys.argv[2]
MIN_AREA = 40
if not os.path.exists(OUT):
    parts = []
    for f in sorted(glob.glob(f"{BD}/*.pkl")):
        name = os.path.basename(f)[:-4]; z = pickle.load(open(f, "rb")); R = pickle.load(open(f"naming_cache/{name}.pkl", "rb"))
        P = z["polys"]; P = P[P.area >= MIN_AREA].reset_index(drop=True)
        cents = ndi.center_of_mass(np.ones_like(R["inst"]), R["inst"], P.poly.values)
        X = PL.rank_pairs(P.poly.values, np.stack(P.feat.values), P.area.values, cents, np.array(z["legend"].index),
                          np.stack(z["legend"].feat.values), z["D"], z["ch"], PL.word_ink(R["rgb"], R["words"]), lab=R["inst"])
        tr = dict(zip(P.poly, P.code)); ar = dict(zip(P.poly, P.area))
        X["zone"] = name; X["vrai"] = X.poly.map(tr); X["aire"] = X.poly.map(ar); X["y"] = (X.code == X.vrai).astype(int)
        parts.append(X); print(name, len(X), flush=True)
    pd.to_pickle(pd.concat(parts, ignore_index=True), OUT)
X = pd.read_pickle(OUT)
tot = X.groupby(["zone", "poly"]).aire.first(); A = {s: tot[tot.index.get_level_values(0).str.startswith(s)].sum() for s in ("south", "north")}
cov = X[X.y == 1].groupby(["zone", "poly"]).aire.first().sum() / tot.sum()
print(f"vrai code parmi les candidats : {cov:.1%} de la surface")
def ev(tr, te, name):
    c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=200, random_state=0)
    c.fit(X.loc[tr, PL.RANK_FEATS], X.loc[tr, "y"]); T = X[te].assign(p=c.predict_proba(X.loc[te, PL.RANK_FEATS])[:, 1])
    B = T.sort_values("p", ascending=False).groupby(["zone", "poly"]).head(1); a = A[name]; out = []
    for t in (0.0, 0.5, 0.8):
        m = B.p >= t; out.append(f"seuil {t}: couv {B.aire[m].sum()/a:5.1%} juste {(B.aire*B.y)[m].sum()/max(B.aire[m].sum(),1):5.1%} total {(B.aire*B.y)[m].sum()/a:5.1%}")
    return " | ".join(out)
S = X.zone.str.startswith("south")
print("SUD->NORD ", ev(S, ~S, "north")); print("NORD->SUD ", ev(~S, S, "south"))
