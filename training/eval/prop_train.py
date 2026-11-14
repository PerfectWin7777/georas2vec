# Entraîne le modèle de décision de la propagation sur les zones NORD vues comme une seule carte.
#   python prop_train.py <sortie.pkl> [zones_exclues séparées par des virgules]
# Validation croisée par groupes de zones (ancres de la zone ET des zones voisines interdites = cas pessimiste).
import sys, glob, pickle, os, numpy as np, pandas as pd
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
import propagate as PR
OUT = sys.argv[1]; EXCL = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()
Ps, Fs = [], []
for f in sorted(glob.glob("prop_cache/north_*.pkl")):
    z = os.path.basename(f)[:-4]
    if z in EXCL:
        continue
    d = pickle.load(open(f, "rb")); P = d["P"].copy(); _, r0, c0 = z.split("_")
    P["zone"] = z; P["zy"], P["zx"] = int(r0), int(c0); P["cy"] += int(r0); P["cx"] += int(c0)
    Ps.append(P); Fs.append(d["F"])
P = pd.concat(Ps, ignore_index=True); F = np.concatenate(Fs)
anchor = np.where((P.source_nom.values == PR.ANCHOR_SRC) & (P.code.values != ""), P.code.values, "")
zy, zx = P.zy.values, P.zx.values
near = lambda I, A: (np.abs(zy[I][:, None] - zy[A][None, :]) <= 2048) & (np.abs(zx[I][:, None] - zx[A][None, :]) <= 2048)
res = {}
for nm, ex in (("normal", None), ("pessimiste", near)):
    X = PR.features(P, F, anchor, exclude=ex)
    X["y"] = (X.c1.values == P.vrai.values[X.i]).astype(int); X["zone"] = P.zone.values[X.i]; X["aire"] = P.aire.values[X.i]
    X["ancre"] = anchor[X.i] != ""; res[nm] = X
    print(nm, len(X), flush=True)
# entraînement : mélange des deux régimes (le modèle voit des cas avec et sans ancres proches)
T = pd.concat([res["normal"], res["pessimiste"]], ignore_index=True); T = T[~T.ancre]
X = res["pessimiste"].copy(); X["p"] = np.nan
for tr, te in GroupKFold(5).split(X, groups=X.zone):
    zt = set(X.zone.iloc[tr]); Tt = T[T.zone.isin(zt)]
    c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
    c.fit(Tt[PR.PFEATS], Tt.y, sample_weight=np.sqrt(Tt.aire)); X.loc[X.index[te], "p"] = c.predict_proba(X.iloc[te][PR.PFEATS])[:, 1]
cand = ~X.ancre
for t in (0.6, 0.7, 0.8, 0.9):
    m = cand & (X.p >= t)
    print(f"pessimiste, seuil {t}: propagés {X.aire[m].sum()/X.aire.sum():.1%} de la surface, justes {X.aire[m & (X.y==1)].sum()/X.aire[m].sum():.1%}")
c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
c.fit(T[PR.PFEATS], T.y, sample_weight=np.sqrt(T.aire))
pickle.dump(dict(model=c, features=PR.PFEATS, thr=0.8, exclues=sorted(EXCL)), open(OUT, "wb"))
print("->", OUT)
