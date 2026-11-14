# Propagation des codes par la couleur / texture de la carte.
# Les polygones dont le code a été LU avec certitude sont des ancres ; chaque autre polygone reçoit le code de l'ancre
# la plus semblable (signature couleur + motif), si un petit modèle appris juge l'attribution sûre.
# Les mêmes caractéristiques servent à l'entraînement (prop_train.py) et dans la chaîne (pipeline.name_polygons).
import numpy as np, pandas as pd

PFEATS = ["d1", "d2", "ratio", "n_anc1", "log_aire", "dist_xy1", "dist_xy_c1", "rank_top_eq", "rank_conf", "dL", "dab", "n_near_c1"]
PFEATS_LEG = PFEATS + ["d_leg1", "leg_gap", "leg_rank"]   # + ressemblance à la case de légende du code proposé
WHITE_L, WHITE_C = 94.0, 4.0
ANCHOR_SRC = "code lu sur la carte"
RADIUS = 2048.0   # px (scans 300 dpi) : échelle des distances et rayon du voisinage


def features(P, F, anchor, exclude=None, min_area=40, chunk=1500, legend_feats=None):
    """P : DataFrame (aire, cy, cx, suggestions, confiance), F : signatures (n x d), anchor : codes des ancres ('' sinon).
    exclude(I, A) -> masque booléen (len(I) x len(A)) d'ancres interdites (tests). Retourne une ligne par polygone candidat."""
    anchor = np.asarray(anchor, dtype=object)
    A = np.nonzero(anchor != "")[0]
    if len(A) == 0:
        return pd.DataFrame(columns=["i", "c1"] + PFEATS)
    codes = anchor[A]; uc = np.unique(codes)
    yx = P[["cy", "cx"]].values.astype(float)
    I_all = np.nonzero(P.aire.values >= min_area)[0]
    out = []
    for s in range(0, len(I_all), chunk):
        I = I_all[s:s + chunk]
        D = np.sqrt(((F[I, None, :] - F[None, A, :]) ** 2).sum(-1))
        D[I[:, None] == A[None, :]] = np.inf
        if exclude is not None:
            D[exclude(I, A)] = np.inf
        Dc = np.full((len(I), len(uc)), np.inf)
        for j, c in enumerate(uc):
            Dc[:, j] = D[:, codes == c].min(1)
        o = np.argsort(Dc, 1)
        for k, i in enumerate(I):
            j1 = o[k, 0]; d1 = Dc[k, j1]
            if not np.isfinite(d1):
                continue
            d2 = Dc[k, o[k, 1]] if len(uc) > 1 else 99.0
            d2 = d2 if np.isfinite(d2) else 99.0
            c1 = uc[j1]; m = codes == c1; ok = np.isfinite(D[k, m])
            Am = A[m][ok]; ba = Am[np.argmin(D[k, m][ok])]
            dd = np.hypot(*(yx[Am] - yx[i]).T)
            out.append((i, c1, d1, d2, d1 / max(d2, 1e-3), len(Am), np.log(P.aire.values[i]),
                        np.hypot(*(yx[ba] - yx[i])) / RADIUS, dd.min() / RADIUS,
                        abs(F[i, 0] - F[ba, 0]), np.hypot(*(F[i, 1:3] - F[ba, 1:3])), int((dd <= RADIUS).sum())))
    X = pd.DataFrame(out, columns=["i", "c1", "d1", "d2", "ratio", "n_anc1", "log_aire", "dist_xy1", "dist_xy_c1", "dL", "dab", "n_near_c1"])
    if len(X):
        top = P.suggestions.fillna("").astype(str).str.split(" ").str[0].values[X.i.values]
        X["rank_top_eq"] = (top == X.c1.values).astype(int)
        X["rank_conf"] = P.confiance.fillna(0).values[X.i.values].astype(float)
        if legend_feats is not None:
            L = {c: np.asarray(v, float) for c, v in legend_feats.items()}
            ucs = [c for c in uc if c in L]; LM = np.stack([L[c] for c in ucs]) if ucs else np.zeros((0, F.shape[1]))
            dl = np.sqrt(((F[X.i.values][:, None, :] - LM[None]) ** 2).sum(-1)) if len(ucs) else np.full((len(X), 1), 99.)
            idx = {c: k for k, c in enumerate(ucs)}
            d1l = np.array([dl[r, idx[c]] if c in idx else 99. for r, c in enumerate(X.c1.values)])
            X["d_leg1"] = d1l; X["leg_gap"] = d1l - dl.min(1)
            X["leg_rank"] = [(dl[r] < d1l[r]).sum() for r in range(len(X))]
    return X


def propagate(P, F, model, thr=None, legend_feats=None):
    """P doit contenir code, source_nom, confiance, suggestions, aire, cy, cx. Les polygones non nommés (et ceux nommés
    seulement par la couleur de la légende, peu fiables sur scan) reçoivent le code propagé si p >= thr."""
    thr = model.get("thr", 0.8) if thr is None else thr
    P = P.copy()
    white = (F[:, 0] > WHITE_L) & (np.hypot(F[:, 1], F[:, 2]) < WHITE_C)   # papier / glace : jamais propagé
    anchor = np.where((P.source_nom.values == ANCHOR_SRC) & (P.code.values != "") & ~white, P.code.values, "")
    X = features(P, F, anchor, legend_feats=legend_feats if "d_leg1" in model["features"] else None)
    if not len(X):
        return P
    X["p"] = model["model"].predict_proba(X[model["features"]])[:, 1]
    water = np.isin(P.source_nom.values, ["eau", "eau ou hors carte"])
    weak = (((P.code.values == "") & ~water) | (P.source_nom.values == "couleur de la légende")) & ~white
    for r in X.itertuples():
        if weak[r.i] and r.p >= thr:
            j = P.index[r.i]
            P.at[j, "code"] = r.c1; P.at[j, "source_nom"] = "même couleur qu'une unité lue sur la carte"
            P.at[j, "confiance"] = round(float(r.p), 2)
        elif P.source_nom.values[r.i] == "couleur de la légende" and r.p < thr:
            j = P.index[r.i]
            P.at[j, "code"] = ""; P.at[j, "source_nom"] = "non identifiée – suggestion probable"
    return P


def clean_anchors(F, anchor, ratio=0.5, min_support=3):
    """Ancres incohérentes : couleur bien plus proche des ancres d'un AUTRE code que des siennes.
    Variante de préfixe d'âge (K vs lK, eKg vs Kg : préfixe perdu à l'OCR) -> code corrigé ; autre code -> ancre retirée.
    Retourne (ancres nettoyées, dict index -> code corrigé)."""
    anchor = np.asarray(anchor, dtype=object).copy()
    A = np.nonzero(anchor != "")[0]
    if len(A) < 3:
        return anchor, {}
    codes = anchor[A]; D = np.sqrt(((F[A, None, :] - F[None, A, :]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    cnt = pd.Series(codes).value_counts()
    fixes, drop = {}, []
    variant = lambda a, b: a != b and (a.endswith(b) or b.endswith(a)) and abs(len(a) - len(b)) <= 2 and min(len(a), len(b)) >= 1
    for k, i in enumerate(A):
        c = codes[k]; same = codes == c
        d_s = D[k, same].min() if same.sum() > 1 else np.inf
        o = ~same
        if not o.any():
            continue
        j = np.argmin(np.where(o, D[k], np.inf)); c2 = codes[j]
        # soutien : nombre d'ancres de c2 plus proches que la plus proche ancre de c
        sup = int(((codes == c2) & (D[k] < d_s)).sum())
        if D[k, j] < ratio * d_s and sup >= min_support:
            if variant(c, c2):
                fixes[i] = c2; anchor[i] = c2
            else:
                drop.append(i)
    anchor[drop] = ""
    return anchor, fixes
