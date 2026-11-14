# =====================================================================================
#  MODULE DE NOMMAGE : à quel polygone appartient chaque code lu sur la carte ?
#  Module séparé du modèle de segmentation. Pour chaque code lu (OCR), on note chaque polygone
#  voisin avec des indices simples (distance, trait de rappel sombre, place disponible pour le
#  texte, recouvrement...) et un petit modèle appris sur de VRAIES étiquettes GMNA choisit.
#  Apprentissage : zones "south" ; test : zones "north" (jamais vues).
# =====================================================================================
import sys, glob, pickle, re, difflib, time
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

RADIUS = 110          # rayon de recherche des polygones candidats autour d'une étiquette (px)
norm = lambda c: re.sub(r"[\*\?\s]", "", str(c))
import os
if os.path.exists("naming_cache/vocab.pkl"):
    _codes = pickle.load(open("naming_cache/vocab.pkl", "rb"))
else:   # lecture encore en cours : vocabulaire = codes des zones déjà lues
    _codes = [c for f in glob.glob("naming_cache/*_*.pkl") for c in pickle.load(open(f, "rb"))["codes"]]
VOCAB = sorted({norm(c) for c in _codes} - {""})
VSET = set(VOCAB)


SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_FLAT = {}


def _flat_keys(code):
    """Formes sous lesquelles l'OCR rend un code à exposants : exposant perdu (X²n -> Xn) ou lu comme chiffre (X2n)."""
    a = re.sub("[⁰¹²³⁴⁵⁶⁷⁸⁹]", "", code); b = code.translate(SUP)
    keys = {a, b}
    # préfixes d'âge en petites capitales sur les cartes (ʟ = late/l, ᴇ = early/e) : l'OCR lit L, I, 1 ou E majuscules
    for k in list(keys):
        if k[:1] == "l" and len(k) > 1:
            keys |= {"L" + k[1:], "I" + k[1:], "1" + k[1:]}
        if k[:1] == "e" and len(k) > 1:
            keys |= {"E" + k[1:]}
    # symbole du Trias (Ʀ, codé TR) : l'OCR lit R, Tr ou k
    for k in list(keys):
        if "TR" in k:
            keys |= {k.replace("TR", "R", 1), k.replace("TR", "Tr", 1)}
    # unité sous-marine 'TQ*' : sur la carte, l'astérisque n'est souvent pas imprimé -> 'TQ' propose aussi 'TQ*' (la couleur tranche)
    for k in list(keys) + [code]:
        if "*" in k:
            keys.add(k.replace("*", ""))
    return {k for k in keys if k and k != code}


_CANON = {}


def canon_lookup(c):
    """Code canonique lu (exposants en chiffres, sans espace) -> code de la légende, s'il existe."""
    if not _CANON:
        for v in VOCAB:
            _CANON.setdefault(str(v).translate(SUP).replace(" ", ""), v)
    return _CANON.get(c) or _CANON.get(c.rstrip("*"))


def set_vocab(codes):
    global VOCAB, VSET, _FLAT
    VOCAB = sorted(codes); VSET = set(VOCAB); _FLAT = {}; _CANON.clear()
    for c in VOCAB:
        for k in _flat_keys(c):
            _FLAT.setdefault(k, set()).add(c)


def to_codes(txt):
    """Liste de (code, qualité) possibles pour un texte lu. Plusieurs codes si l'exposant est perdu : la légende tranche."""
    if not _FLAT or set().union(*_FLAT.values()) - VSET:
        set_vocab(VOCAB)
    raw = re.sub(r"[^A-Za-z0-9€Є⁰¹²³⁴⁵⁶⁷⁸⁹]", "", txt).replace("€", "Є")
    if not raw:
        return []
    out = {raw: 1.0} if raw in VSET else {}
    variants = [raw] + [raw[0] + v for v in {raw[1:].replace("l", "1"), raw[1:].replace("z", "2"), raw[1:].replace("l", "1").replace("z", "2")} if v != raw[1:]]
    hits = set()
    for v in variants:
        hits |= _FLAT.get(v, set())
    for c in hits - set(out):
        out[c] = 0.9 if len(hits) == 1 and not out else 0.85
    if not out:   # approché : sur les codes ET leurs formes sans exposant
        keys = VOCAB + list(_FLAT)
        m = difflib.get_close_matches(raw, keys, n=1, cutoff=0.75)
        if m:
            for c in ([m[0]] if m[0] in VSET else sorted(_FLAT[m[0]])):
                out[c] = 0.75
    return sorted(out.items(), key=lambda x: -x[1])


def to_code(txt):
    r = to_codes(txt)
    return r[0] if r else (None, 0.0)


FEATS = ["dist_centre", "dist_boite", "sous_centre", "part_boite", "rang_dist", "n_cand", "aire_log", "rayon_int",
         "rayon_sur_texte", "trait_sombre", "trait_lum", "limites_croisees", "long_trait", "h_texte", "l_texte",
         "conf", "qualite_code", "dist_rel", "plus_proche_non_sous", "part_boite_max_autre", "vise_trait", "trait_L"]


def candidates(reg, wd):
    """Lignes de caractéristiques (une par polygone candidat) pour un mot lu."""
    inst, lum = reg["inst"], reg["lum"]
    N = inst.shape[0]
    x0, y0, x1, y1 = int(max(wd.x0, 0)), int(max(wd.y0, 0)), int(min(wd.x1, N - 1)), int(min(wd.y1, N - 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return []
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0, X1, Y1 = max(x0 - RADIUS, 0), max(y0 - RADIUS, 0), min(x1 + RADIUS, N - 1), min(y1 + RADIUS, N - 1)
    sub, slum = inst[Y0:Y1 + 1, X0:X1 + 1], lum[Y0:Y1 + 1, X0:X1 + 1]
    yy, xx = np.mgrid[Y0:Y1 + 1, X0:X1 + 1]
    dc = np.hypot(yy - cy, xx - cx)                                           # distance au centre de la boîte
    db = np.hypot(np.maximum(np.maximum(y0 - yy, yy - y1), 0), np.maximum(np.maximum(x0 - xx, xx - x1), 0))  # à la boîte
    ids = np.unique(sub); ids = ids[ids > 0]
    if not len(ids):
        return []
    box = sub[y0 - Y0:y1 - Y0 + 1, x0 - X0:x1 - X0 + 1]
    light = slum[y0 - Y0:y1 - Y0 + 1, x0 - X0:x1 - X0 + 1] >= 120            # hors encre du texte
    nb = max(light.sum(), 1)
    under = inst[int(cy), int(cx)]
    th, tw = y1 - y0, x1 - x0
    dmin_c = ndi.minimum(dc, sub, ids); dmin_b = ndi.minimum(db, sub, ids)
    pos_b = ndi.minimum_position(db, sub, ids)
    edt = ndi.distance_transform_edt(sub > 0) if False else None
    from leader import trace
    tips = trace(reg, wd)
    rows = []
    order = np.argsort(dmin_b)
    rank = np.empty(len(ids)); rank[order] = np.arange(len(ids))
    part = np.array([(box[light] == i).sum() / nb for i in ids])
    for j, i in enumerate(ids):
        m = sub == i
        # rayon intérieur du polygone (dans la fenêtre) : peut-il contenir le texte ?
        rin = ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1].max()
        # segment droit boîte -> point le plus proche du polygone : un trait de rappel est sombre et droit
        py, px = pos_b[j][0] + Y0, pos_b[j][1] + X0
        qy, qx = min(max(py, y0), y1), min(max(px, x0), x1)
        L = float(np.hypot(py - qy, px - qx))
        if L >= 3:
            n = int(L) + 1
            sy = np.clip(np.round(np.linspace(qy, py, n)).astype(int), 0, N - 1)
            sx = np.clip(np.round(np.linspace(qx, px, n)).astype(int), 0, N - 1)
            ls = np.minimum.reduce([lum[np.clip(sy + a, 0, N - 1), np.clip(sx + b, 0, N - 1)] for a in (-1, 0, 1) for b in (-1, 0, 1)])
            dark, lmean = float((ls[1:-1] < 120).mean()) if n > 2 else 0.0, float(ls.mean())
            cross = int((np.diff(inst[sy, sx]) != 0).sum())
        else:
            dark, lmean, cross = 0.0, 255.0, 0
        others = np.delete(part, j)
        tl = [t[2] for t in tips if inst[t[0], t[1]] == i]
        rows.append(dict(poly=int(i), dist_centre=float(dmin_c[j]), dist_boite=float(dmin_b[j]), sous_centre=float(i == under),
                         part_boite=float(part[j]), rang_dist=float(rank[j]), n_cand=float(len(ids)),
                         aire_log=float(np.log1p(reg["area"][i])), rayon_int=float(rin), rayon_sur_texte=float(rin / max(th / 2, 1)),
                         trait_sombre=dark, trait_lum=lmean, limites_croisees=float(cross), long_trait=L, h_texte=float(th),
                         l_texte=float(tw), conf=float(wd.conf), qualite_code=float(wd.q),
                         dist_rel=float(dmin_b[j] / max(th, 1)), plus_proche_non_sous=float(rank[j] == (1 if under in ids else 0) and i != under),
                         part_boite_max_autre=float(others.max() if len(others) else 0),
                         vise_trait=float(len(tl)), trait_L=float(min(tl)) if tl else -1.0))
    return rows


def load_region(f):
    r = pickle.load(open(f, "rb"))
    r["lum"] = (r["rgb"].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
    r["area"] = np.bincount(r["inst"].ravel())
    r["true"] = {i + 1: norm(c) for i, c in enumerate(r["codes"])}
    return r


def build(files):
    out = []
    for f in files:
        reg = load_region(f)
        W_ = reg["words"]
        for k, wd in enumerate(W_.itertuples()):
            code, q = to_code(wd.text)
            if code is None:
                continue
            wd = wd._replace(conf=wd.conf) if False else wd
            class _W: pass
            w = _W(); w.__dict__.update(wd._asdict()); w.q = q
            for row in candidates(reg, w):
                row.update(region=f, word=k, code=code, y=float(reg["true"][row["poly"]] == code),
                           vrai=reg["true"][row["poly"]], aire=float(reg["area"][row["poly"]]))
                out.append(row)
    return pd.DataFrame(out)


def assign(D, score):
    """Choix final : pour chaque mot le polygone le mieux noté ; un polygone ne garde qu'UN code
    (celui qui a la meilleure note), les mots perdants passent à leur candidat suivant."""
    D = D.assign(s=score)
    choice = {}
    words = {k: g.sort_values("s", ascending=False) for k, g in D.groupby(["region", "word"])}
    ptr = {k: 0 for k in words}
    for _ in range(4):
        for k, g in words.items():
            if ptr[k] < len(g):
                r = g.iloc[ptr[k]]
                choice[k] = (r.region, int(r.poly), r.code, float(r.s))
            else:
                choice.pop(k, None)
        best = {}
        for k, (reg, p, c, s) in choice.items():
            key = (reg, p)
            if key not in best or s > best[key][1]:
                best[key] = (c, s)
        moved = False
        for k, (reg, p, c, s) in list(choice.items()):
            if best[(reg, p)][0] != c and ptr[k] + 1 < len(words[k]) and words[k].iloc[ptr[k] + 1].s > 0.15:
                ptr[k] += 1; moved = True
        if not moved:
            break
    return choice


def report(D, choice, title):
    key = D.set_index(["region", "word", "poly"])
    evaluable = D.groupby(["region", "word"]).y.max() > 0          # le bon code existe parmi les candidats (lecture juste)
    ok = n = 0
    polys = {}
    for k, (reg, p, c, s) in choice.items():
        r = key.loc[(k[0], k[1], p)]
        good = bool(np.atleast_1d(r.y)[0])
        if evaluable[k]:
            n += 1; ok += good
        cur = polys.get((reg, p))
        if cur is None or s > cur[1]:
            polys[(reg, p)] = (good, s, float(np.atleast_1d(r.aire)[0]))
    pg = np.array([v[0] for v in polys.values()]); pa = np.array([v[2] for v in polys.values()])
    print(f"{title:<46} codes bien rattachés : {ok / max(n, 1):5.1%} ({n} codes) | polygones nommés : {len(polys)}, "
          f"noms justes : {pg.mean():5.1%} (en surface : {(pa * pg).sum() / pa.sum():5.1%})")
    return ok / max(n, 1), polys


if __name__ == "__main__":
    t0 = time.time()
    tr_files, te_files = sorted(glob.glob("naming_cache/south_*.pkl")), sorted(glob.glob("naming_cache/north_*.pkl"))
    if len(te_files) < 3:   # provisoire : pas encore de zones nord -> 1 zone sud sur 3 en test
        allf = tr_files + te_files
        tr_files, te_files = [f for i, f in enumerate(allf) if i % 3], [f for i, f in enumerate(allf) if i % 3 == 0]
    Dtr, Dte = build(tr_files), build(te_files)
    print(f"apprentissage : {len(tr_files)} zones, {Dtr.groupby(['region', 'word']).ngroups} codes lus, {len(Dtr)} paires | "
          f"test : {len(te_files)} zones, {Dte.groupby(['region', 'word']).ngroups} codes lus  ({time.time() - t0:.0f} s)")
    ev = Dtr.groupby(["region", "word"]).y.transform("max") > 0    # on n'apprend que sur les mots correctement lus
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=24, l2_regularization=1.0,
                                         class_weight="balanced", random_state=0)
    clf.fit(Dtr.loc[ev, FEATS], Dtr.loc[ev, "y"])
    pickle.dump({"model": clf, "features": FEATS, "radius": RADIUS}, open("naming_model.pkl", "wb"))
    print("\nSUR LES ZONES DE TEST (north, jamais vues) :")
    naive = Dte.sous_centre + 0.001 * Dte.part_boite
    a0, _ = report(Dte, {k: (g.region.iloc[0], int(g.poly.iloc[np.argmax(n_)]), g.code.iloc[0], 1.0)
                         for (k, g), n_ in ((kg, naive.loc[kg[1].index].values) for kg in Dte.groupby(["region", "word"]))},
                   "Règle naïve (polygone sous l'étiquette)")
    p = clf.predict_proba(Dte[FEATS])[:, 1]
    a1, _ = report(Dte, {k: (g.region.iloc[0], int(g.poly.iloc[np.argmax(p[Dte.index.get_indexer(g.index)])]), g.code.iloc[0],
                             float(p[Dte.index.get_indexer(g.index)].max())) for k, g in Dte.groupby(["region", "word"])},
                   "Module appris (meilleure note)")
    ch = assign(Dte, p)
    a2, polys = report(Dte, ch, "Module appris + un seul code par polygone")
    # avec seuil de confiance : on ne nomme que si la note est élevée
    for thr in (0.5, 0.7, 0.85):
        report(Dte, {k: v for k, v in ch.items() if v[3] >= thr}, f"  ... en ne gardant que les notes >= {thr}")
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(clf, Dte[FEATS], Dte["y"], n_repeats=3, random_state=0, scoring="average_precision")
    print("\nIndices les plus utiles :", ", ".join(f"{FEATS[i]} ({imp.importances_mean[i]:.3f})" for i in np.argsort(-imp.importances_mean)[:8]))
