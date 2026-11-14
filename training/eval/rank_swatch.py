import os as _o, sys as _s
_R = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..") if "eval" in __file__ else _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..")
_s.path[:0] = [_o.path.join(_R, "georas2vec"), _o.path.join(_R, "training", "eval"), _o.path.join(_R, "training")]
# Classeur de nommage + comparateur de cases appris (couleur + motif), avec la VRAIE légende imprimée.
# Ajoute aux paires (polygone, code) : sw_sim (ressemblance), sw_rank (rang de la case parmi toutes), sw_gap (écart au meilleur).
# Encodeur entraîné sur la feuille NORD seulement -> mesure honnête sur le SUD ; sens inverse non testable (encodeur vu le nord).
import os, glob, pickle, numpy as np, pandas as pd, torch
from PIL import Image
from scipy import ndimage as ndi
import swatch_match as SM, code_reader as CR
from naming_v08_test import canon
torch.set_num_threads(4)
enc = SM.Enc(); enc.load_state_dict(torch.load(os.environ.get("ENC", "swatch_enc.pt"), weights_only=True)); enc.eval()
SW = SM.legend_swatches(); swc = sorted(SW)
with torch.no_grad():
    ES = enc(SM.tens(np.stack([SW[c] for c in swc]))).numpy()
SWI = {c: i for i, c in enumerate(swc)}


def poly_embeddings(rgb, inst, ids, n=3, seed=0):
    rng = np.random.default_rng(seed); dt = ndi.distance_transform_edt(inst > 0)
    out = np.zeros((len(ids), ES.shape[1]), np.float32)
    patches, owner = [], []
    for k, i in enumerate(ids):
        ys, xs = np.nonzero(inst == i)
        if not len(ys):
            continue
        pick = rng.choice(len(ys), min(n, len(ys)), replace=False)
        for j in pick:
            y, x = ys[j], xs[j]; y0 = int(np.clip(y - SM.PH, 0, inst.shape[0] - 2 * SM.PH)); x0 = int(np.clip(x - SM.PW, 0, inst.shape[1] - 2 * SM.PW))
            patches.append(np.asarray(Image.fromarray(rgb[y0:y0 + 2 * SM.PH, x0:x0 + 2 * SM.PW]).resize((SM.PW, SM.PH), Image.BILINEAR))); owner.append(k)
    owner = np.array(owner)
    with torch.no_grad():
        E = np.concatenate([enc(SM.tens(np.stack(patches[s:s + 512]))).numpy() for s in range(0, len(patches), 512)])
    for k in range(len(ids)):
        m = owner == k
        if m.any():
            v = E[m].mean(0); out[k] = v / (np.linalg.norm(v) + 1e-9)
    return out


if __name__ == "__main__":
    parts = []
    for f in sorted(glob.glob("bench_auto/*.pkl")):
        name = os.path.basename(f)[:-4]; out = f"bench_sw/{name}.pkl"; os.makedirs("bench_sw", exist_ok=True)
        if os.path.exists(out):
            parts.append(pd.read_pickle(out)); continue
        X = pd.read_pickle(f); R = pickle.load(open(f"naming_cache/{name}.pkl", "rb"))
        ids = np.unique(X.poly.values); E = poly_embeddings(R["rgb"], R["inst"], ids)
        S = E @ ES.T                                                      # polygone x case
        rank = np.argsort(np.argsort(-S, 1), 1); best = S.max(1)
        pi = {p: k for k, p in enumerate(ids)}
        k = X.poly.map(pi).values; j = X.code.map(lambda c: SWI.get(c, -1)).values
        X["sw_sim"] = np.where(j >= 0, S[k, np.maximum(j, 0)], -1.0)
        X["sw_rank"] = np.where(j >= 0, rank[k, np.maximum(j, 0)], 999)
        X["sw_gap"] = np.where(j >= 0, best[k] - S[k, np.maximum(j, 0)], 2.0)
        X.to_pickle(out); parts.append(X); print(name, flush=True)
    pd.concat(parts, ignore_index=True).to_pickle("rank_sw.pkl")
    # évaluation : classeur entraîné sur le NORD, testé sur le SUD (l'encodeur n'a vu que le nord)
    import sys; sys.path.insert(0, "."); import pipeline as PL
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = pd.read_pickle("rank_sw.pkl"); S_ = X.zone.str.startswith("south")
    A = X[S_].groupby(["zone", "poly"]).aire.first().sum()
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=200, random_state=0)
    BASE = PL.RANK_FEATS + PL.INK_FEATS + PL.INSIDE_FEATS; SWF = ["sw_sim", "sw_rank", "sw_gap"]
    for nm, FE in (("vraie légende, sans comparateur", BASE), ("vraie légende + comparateur appris", BASE + SWF)):
        c = mk(); c.fit(X.loc[~S_, FE], X.loc[~S_, "y"]); T = X[S_].assign(p=c.predict_proba(X.loc[S_, FE])[:, 1])
        B = T.sort_values("p", ascending=False).groupby(["zone", "poly"]).head(1)
        out = [f"1re suggestion juste {(B.aire*B.y).sum()/A:5.1%}"]
        for t in (0.5, 0.7, 0.8):
            m = B.p >= t
            out.append(f"seuil {t}: nommée {B.aire[m].sum()/A:5.1%} juste {(B.aire*B.y)[m].sum()/max(B.aire[m].sum(),1):5.1%} total {(B.aire*B.y)[m].sum()/A:5.1%}")
        print(f"{nm:<36} NORD->SUD: " + " | ".join(out), flush=True)
    c = mk(); c.fit(X[BASE + SWF], X.y); pickle.dump(dict(model=c, features=BASE + SWF, K=12, sure=0.8), open("legend_ranker_sw.pkl", "wb"))
