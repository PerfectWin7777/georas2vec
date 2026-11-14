# Banc eau / hors carte (expérience). Le code produit est dans water.py.
import sys, os, glob, pickle, numpy as np, pandas as pd
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import naming_model as NM
from water import WFEATS, region_feats
from sklearn.ensemble import HistGradientBoostingClassifier

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists("water_data.pkl"):
        NM.set_vocab(NM.VOCAB)
        parts = []
        for f in sorted(glob.glob("naming_cache/north_*.pkl") + glob.glob("naming_cache/south_*.pkl")):
            R = pickle.load(open(f, "rb")); inst = R["inst"]
            holes, n = ndi.label(inst == 0)
            reg = np.where(inst > 0, inst, holes + inst.max())
            D = region_feats(R["rgb"], reg, R["words"])
            D["eau"] = (D.poly > inst.max()).astype(int); D["zone"] = os.path.basename(f)[:-4]
            parts.append(D); print(f, len(D), int(D.eau.sum()), flush=True)
        pd.to_pickle(pd.concat(parts, ignore_index=True), "water_data.pkl")
    X = pd.read_pickle("water_data.pkl"); X["aire"] = np.exp(X.log_aire)
    S = X.zone.str.startswith("south")
    print("part de surface eau/hors carte : SUD %.1f%%  NORD %.1f%%" % (100 * (X.aire * X.eau)[S].sum() / X.aire[S].sum(), 100 * (X.aire * X.eau)[~S].sum() / X.aire[~S].sum()))
    for tr, te, nm in ((S, ~S, "SUD->NORD"), (~S, S, "NORD->SUD")):
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, random_state=0)
        c.fit(X.loc[tr, WFEATS], X.loc[tr, "eau"], sample_weight=np.sqrt(X.aire[tr]))
        p = c.predict_proba(X.loc[te, WFEATS])[:, 1]; T = X[te]
        for t in (0.5, 0.7, 0.9):
            m = p >= t
            prec = (T.aire * T.eau)[m].sum() / max(T.aire[m].sum(), 1); rec = (T.aire * T.eau)[m].sum() / (T.aire * T.eau).sum()
            print(f"{nm} seuil {t}: eau détectée (surface) {rec:.1%} | précision {prec:.1%} | unités prises pour de l'eau {(T.aire * (1 - T.eau))[m].sum() / (T.aire * (1 - T.eau)).sum():.2%} de leur surface")
    c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, random_state=0)
    c.fit(X[WFEATS], X.eau, sample_weight=np.sqrt(X.aire))
    pickle.dump(dict(model=c, features=WFEATS), open("water_model.pkl", "wb"))
