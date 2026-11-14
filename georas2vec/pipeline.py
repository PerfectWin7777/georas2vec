# =====================================================================================
#  GEORAS2VEC – PIPELINE COMPLET : raster de carte géologique -> GeoPackage QGIS / ArcGIS
#   1. modèle (affinités + failles) par fenêtres glissantes   2. mutex watershed -> polygones
#   3. vectorisation (unités + failles)                      4. OCR des textes de la carte
#   5. nommage : code écrit (module appris) > case de légende (si sûr) > "Unité non identifiée" + 3 suggestions
#   6. export GeoPackage (couches unites_geologiques, failles, table legende + styles QGIS par défaut)
# =====================================================================================
import os, sys, re, sqlite3, pickle, time
import numpy as np, pandas as pd, torch, geopandas as gpd
from scipy import ndimage as ndi
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape, LineString
from skimage.color import rgb2lab as sk_rgb2lab
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gc_model as M


# ------------------------------------------------------------------ 1. modèle
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    M.CFG.update(cfg); M.OFFSETS[:] = cfg["offsets"]
    model = M.GeoRas2VecNet(cfg, pretrained=False)
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return model.eval()


@torch.no_grad()
def _run(model, im):
    x = torch.from_numpy(np.ascontiguousarray(im).transpose(2, 0, 1).copy())[None]
    return torch.sigmoid(model(x).float())[0].numpy()


def _tta(model, im, offsets):
    """Moyenne sur 4 miroirs ; chaque décalage est ramené dans le repère d'origine (testé : exact)."""
    H, W = im.shape[:2]
    Y, X = np.mgrid[:H, :W]
    idx = {tuple(o): k for k, o in enumerate(offsets)}
    K = len(offsets)
    acc = 0
    for fy in (0, 1):
        for fx in (0, 1):
            v = im[::-1] if fy else im
            v = v[:, ::-1] if fx else v
            o = _run(model, v)
            A, pf = o[:K], o[K]
            py, px = (H - 1 - Y) if fy else Y, (W - 1 - X) if fx else X
            out = np.empty_like(o)
            for k, (dy, dx) in enumerate(offsets):
                t = (-dy if fy else dy, -dx if fx else dx)
                if t in idx:
                    out[k] = A[idx[t]][py, px]
                else:
                    out[k] = A[idx[(-t[0], -t[1])]][np.clip(py + t[0], 0, H - 1), np.clip(px + t[1], 0, W - 1)]
            out[K] = pf[py, px]
            acc = acc + out
    return acc / 4


def infer(model, rgb, tile=512, stride=384, tta=False):
    """Fenêtres glissantes avec pondération de Hann -> affinités (K x H x W) + probabilité de faille."""
    H, W = rgb.shape[:2]
    ph, pw = max(tile - H, 0), max(tile - W, 0)
    img = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
    Hp, Wp = img.shape[:2]
    K = len(M.OFFSETS)
    acc = np.zeros((K + 1, Hp, Wp), np.float32); ws = np.zeros((Hp, Wp), np.float32)
    hw = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), 1e-3).astype(np.float32)
    pos = lambda n: sorted(set(list(range(0, n - tile + 1, stride)) + [n - tile]))
    for y in pos(Hp):
        for x in pos(Wp):
            im = img[y:y + tile, x:x + tile]
            o = _tta(model, im, M.OFFSETS) if tta else _run(model, im)
            acc[:, y:y + tile, x:x + tile] += o * hw; ws[y:y + tile, x:x + tile] += hw
    o = acc / ws
    return o[:K, :H, :W], o[K, :H, :W]


# ------------------------------------------------------------------ 3. vectorisation
def mws(aff, min_size=20, beta=1.0):
    """Mutex watershed ; beta > 1 renforce la séparation (utile sur scans réels : petites +4 pts, mais grandes synthétiques -1,6 pt)."""
    import seg_eval as SE
    return SE.mws(aff, min_size=min_size, beta=beta)


SMOOTH_PX = 1.0   # validé : -54 % de sommets, fidélité des limites à 2 px inchangée, couverture sans trou ni chevauchement


def vectorize_units(lab, transform, crs, smooth_px=None):
    """Polygones des régions, lissés ENSEMBLE (limites communes simplifiées une seule fois : aucun vide ni chevauchement)."""
    import shapely
    from shapely.affinity import affine_transform
    smooth_px = SMOOTH_PX if smooth_px is None else smooth_px
    geoms = [shape(g) for g, v in shapes(lab.astype(np.int32), mask=lab > 0)]
    lines = shapely.node(shapely.union_all(shapely.boundary(np.array(geoms, dtype=object))))
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(lines)))
    xy = shapely.get_coordinates(shapely.point_on_surface(faces))
    vals = lab[np.clip(xy[:, 1].astype(int), 0, lab.shape[0] - 1), np.clip(xy[:, 0].astype(int), 0, lab.shape[1] - 1)]
    keep = vals > 0; faces, vals = faces[keep], vals[keep]
    if smooth_px > 0:
        faces = shapely.coverage_simplify(faces, smooth_px, simplify_boundary=False)
    m = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
    G = gpd.GeoDataFrame({"pid": vals.astype(int)}, geometry=[affine_transform(f, m) for f in faces], crs=crs)
    P = G.dissolve("pid").reset_index()
    P["geometry"] = P.geometry.make_valid()
    return P


# réglages validés (banc de validation, confirmés sur le test) : seuil failles 0.2 (F1 réel 0.63 -> 0.76), région minimale 20 px
FAULT_THR, MWS_MIN, MWS_BETA = 0.2, 20, 1.0

INFER_SCALE = float(os.environ.get("GEORAS2VEC_SCALE", 1.25))   # échelle d'agrandissement (petits polygones)
INFER_FUSE = os.environ.get("GEORAS2VEC_FUSE", "1") == "1"         # fusion échelle 1 + échelle agrandie (voir JOURNAL v0.4)


def segment(model, rgb, tta=False, scale=None, fuse=None):
    """Segmentation finale. Par défaut : intersection des régions de l'échelle 1 (sépare mieux les grandes unités) et de
    l'échelle agrandie (trouve les petites), éclats < MWS_MIN px rattachés au voisin. Probabilité de faille : échelle 1.
    Test officiel, scans réels : grandes 93,4 -> 94,5 %, petites 74,8 -> 77,0 %, surface 98,3 -> 98,5 %."""
    import seg_eval as SE
    fuse = INFER_FUSE if fuse is None else fuse
    s = INFER_SCALE if scale is None else scale
    if not fuse or s == 1:
        return segment_scale(model, rgb, tta, s)
    a, pf = segment_scale(model, rgb, tta, 1.0)
    b, _ = segment_scale(model, rgb, tta, s)
    _, inv = np.unique(a.astype(np.int64) * (int(b.max()) + 1) + b, return_inverse=True)
    return SE.remove_small(inv.reshape(a.shape).astype(np.int32) + 1, MWS_MIN), pf


def segment_scale(model, rgb, tta=False, scale=None):
    """Affinités sur l'image agrandie x scale -> MWS -> retour à la résolution d'origine (plus proche voisin).
    La probabilité de faille est ramenée à la résolution d'origine par moyenne (interpolation linéaire anti-crénelage)."""
    from skimage.transform import resize
    import seg_eval as SE
    s = INFER_SCALE if scale is None else scale
    H, W = rgb.shape[:2]
    if s == 1:
        aff, pf = infer(model, rgb, tta=tta)
        return mws(aff, MWS_MIN, MWS_BETA), pf
    big = (resize(rgb, (round(H * s), round(W * s)), order=1, anti_aliasing=False, preserve_range=True)).astype(np.uint8)
    aff, pf = infer(model, big, tta=tta); del big
    lab = mws(aff, int(MWS_MIN * s * s), MWS_BETA); del aff
    yi = np.minimum((np.arange(H) * s + s / 2).astype(int), lab.shape[0] - 1)
    xi = np.minimum((np.arange(W) * s + s / 2).astype(int), lab.shape[1] - 1)
    lab = SE.remove_small(lab[yi][:, xi], MWS_MIN)
    pf = resize(pf, (H, W), order=1, anti_aliasing=True).astype(np.float32)
    return lab, pf


FAULT_LINE_THR = 0.05   # choisi sur la validation (scans réels F1 0.75 ; test 0.80 ; synthétique 0.88)


def vectorize_faults(pf, transform, crs, thr=None, min_px=8):
    """Failles : squelette de la carte de probabilité (petites coupures refermées), découpé aux jonctions,
    chaque tronçon ordonné puis raccroché à ses jonctions."""
    import fault_seg as FS
    thr = FAULT_LINE_THR if thr is None else thr
    lines = []
    for path in FS.segments(pf, thr=thr, close=1, min_px=min_px):
        if len(path) >= 2:
            lines.append(LineString([transform * (x + 0.5, y + 0.5) for y, x in path]).simplify(abs(transform.a)))
    F = gpd.GeoDataFrame({"type": ["indéterminée"] * len(lines)}, geometry=lines, crs=crs)
    F["longueur_km"] = (F.length / 1000).round(2)
    return F


# ------------------------------------------------------------------ 4. OCR
OCR_ALLOW = os.environ.get("GEORAS2VEC_OCR_ALLOW", "1") == "1"   # lecture limitée aux caractères des codes de la légende


def code_alphabet(codes):
    import naming_model as NM
    return "".join(sorted({ch for c in codes for ch in str(c).translate(NM.SUP) if ch.isalnum()}))


def read_text(rgb, reader, B=1024, OV=96, allow=None):
    words = []
    kw = dict(allowlist=allow) if allow else {}
    H, W = rgb.shape[:2]
    for y in range(0, H, B - OV):
        for x in range(0, W, B - OV):
            for box, txt, conf in reader.readtext(rgb[y:y + B, x:x + B], mag_ratio=2.0, text_threshold=0.6, low_text=0.3, **kw):
                b = np.array(box) + [x, y]; cx, cy = b.mean(0)
                if (x > 0 and cx < x + OV / 2) or (y > 0 and cy < y + OV / 2):
                    continue
                words.append(dict(text=txt, conf=conf, cx=cx, cy=cy, x0=b[:, 0].min(), x1=b[:, 0].max(), y0=b[:, 1].min(), y1=b[:, 1].max()))
    return pd.DataFrame(words, columns=["text", "conf", "cx", "cy", "x0", "x1", "y0", "y1"])


# ------------------------------------------------------------------ 5. nommage
def swatch_features(lab_pixels):
    """Signature d'une case de légende / d'un polygone : couleur dominante, couleurs claire et foncée du motif, texture."""
    px = lab_pixels[lab_pixels[:, 0] > 45]
    if len(px) < 20:
        px = lab_pixels
    L = px[:, 0]; med = np.percentile(L, 50)
    return np.concatenate([np.median(px, 0), np.median(px[L >= med], 0), np.median(px[L <= med], 0), px.std(0) / 4])


def ocr_candidates(rgb, lab, words, vocab, naming_pkl, flt=None):
    """Toutes les paires (code lu, polygone candidat) avec leur score appris ; et l'affectation un code -> un polygone."""
    import naming_model as NM
    NM.set_vocab(vocab)
    mdl = pickle.load(open(naming_pkl, "rb"))
    reg = dict(inst=lab, rgb=rgb, lum=rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32),
               area=np.bincount(lab.ravel()), flt=flt)
    rows = []
    for k, wd in enumerate(words.itertuples()):
        cq = NM.to_codes(wd.text)
        if not cq:
            continue
        class _W: pass
        w = _W(); w.__dict__.update(wd._asdict()); w.q = cq[0][1]
        cand = NM.candidates(reg, w)
        for code, q in cq:      # un mot ambigu (exposant perdu) donne une ligne par code possible
            for r in cand:
                r = dict(r, qualite_code=q); r.update(region="carte", word=k, code=code, lu=wd.text); rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["word", "code", "s"]), {}
    D = pd.DataFrame(rows)
    D["s"] = mdl["model"].predict_proba(D[mdl["features"]])[:, 1]
    return D, NM.assign(D, D["s"].values)

RANK_FEATS = ["d_norm", "d_rel", "rang", "gap12", "d_ratio", "ocr", "ocr_assigne", "ocr_rel", "lu", "n_lu", "dist_lu", "log_aire",
              "q_code", "n_alt"]

def rank_pairs(pids, feats, areas, cents, codes, Fm, D, ch, words, K=12):
    """Paires (polygone, code candidat) et leurs indices. Candidats = K cases de légende les plus proches + codes lus par l'OCR.
    Indices : uniquement le texte lu sur la carte et la légende (pas de propagation d'apparence entre polygones)."""
    cidx = {c: i for i, c in enumerate(codes)}
    d = np.linalg.norm(feats[:, None, :] - Fm[None], axis=2)
    E, lu_pos, asg = {}, {}, {}
    if len(D):
        for (p, c), s in D.groupby(["poly", "code"]).s.max().items():
            E[(p, c)] = s
        for w, c in D[["word", "code"]].drop_duplicates().values:
            lu_pos.setdefault(c, []).append((words.iloc[w].cy, words.iloc[w].cx))
    for (_, w), (_, p, c, s) in ch.items():
        asg[(p, c)] = max(asg.get((p, c), 0), s)
        if len(D):   # mot ambigu : tous ses codes possibles sur ce polygone sont "affectés"
            for c2 in D.loc[(D.word == w) & (D.poly == p), "code"].unique():
                asg[(p, c2)] = max(asg.get((p, c2), 0), s)
    Ep = {}
    for (p, c), s in E.items():
        Ep.setdefault(p, []).append(c)
    lu_pos = {c: np.array(v) for c, v in lu_pos.items()}
    rows = []
    for i, p in enumerate(pids):
        o = np.argsort(d[i]); rank = np.empty(len(codes), int); rank[o] = np.arange(len(codes))
        cand = list(dict.fromkeys(list(codes[o[:K]]) + [c for c in Ep.get(p, []) if c in cidx]))
        emax = max([E[(p, c)] for c in Ep.get(p, [])] + [0]); gap = d[i, o[1]] - d[i, o[0]] if len(o) > 1 else 0
        for c in cand:
            j = cidx[c]; pos = lu_pos.get(c)
            rows.append(dict(poly=p, code=c, d=d[i, j], d_rel=d[i, j] - d[i, o[0]], rang=rank[j], gap12=gap,
                             ocr=E.get((p, c), 0), ocr_assigne=asg.get((p, c), 0), ocr_max_poly=emax,
                             lu=int(pos is not None), n_lu=0 if pos is None else len(pos),
                             dist_lu=5000 if pos is None else np.hypot(pos[:, 0] - cents[i][0], pos[:, 1] - cents[i][1]).min(),
                             log_aire=np.log(areas[i])))
    X = pd.DataFrame(rows)
    X["d_ratio"] = X.d_rel / (X.gap12 + 1); X["d_norm"] = X.d / X.d.median(); X["ocr_rel"] = X.ocr - X.ocr_max_poly
    # ambiguïté de lecture : qualité de la correspondance texte -> code, nombre de codes possibles pour le mot (exposant perdu)
    if len(D) and "qualite_code" in D:
        nalt = D.groupby("word").code.nunique().rename("nalt")
        a = D.join(nalt, on="word").groupby(["poly", "code"]).agg(q_code=("qualite_code", "max"), n_alt=("nalt", "min")).reset_index()
        X = X.merge(a, on=["poly", "code"], how="left")
    for c in ("q_code", "n_alt"):
        X[c] = X[c].fillna(0) if c in X else 0.0
    return X

def name_polygons(rgb, lab, words, legend, naming_pkl, flt=None, ranker_pkl="legend_ranker.pkl", sure=None, check=0.5):
    """legend : DataFrame index=code, colonnes nom, age, lithologie, couleur_hex, feat (signature de la case).
    Retourne par polygone : code, source_nom, confiance, texte_lu, suggestions.
    source_nom : 'code lu sur la carte' / 'légende' (confiance >= sure), '… – à vérifier' (check <= confiance < sure), 'non identifiée'."""
    D, ch = ocr_candidates(rgb, lab, words, legend.index, naming_pkl, flt)
    R = pickle.load(open(ranker_pkl, "rb"))
    sure = sure if sure is not None else R.get("sure", 0.8)
    labc = sk_rgb2lab(rgb).reshape(-1, 3)
    ids = np.unique(lab[lab > 0])
    order = np.argsort(lab.ravel(), kind="stable"); sl = lab.ravel()[order]
    starts = np.searchsorted(sl, ids); ends = np.searchsorted(sl, ids, side="right")
    feats = np.stack([swatch_features(labc[order[a:b]]) for a, b in zip(starts, ends)])
    cents = ndi.center_of_mass(np.ones_like(lab), lab, ids)
    codes = np.array(legend.index); Fm = np.stack(legend.feat.values)
    X = rank_pairs(ids, feats, ends - starts, cents, codes, Fm, D, ch, words, K=R.get("K", 12))
    X["p"] = R["model"].predict_proba(X[R["features"]])[:, 1]
    X = X.sort_values(["poly", "p"], ascending=[True, False])
    lu_txt = {}
    for (_, w), (_, p, c, s) in ch.items():
        lu_txt[(p, c)] = words.iloc[w].text
    out = {}
    for p, g in X.groupby("poly", sort=False):
        top = g.head(3); c, pr = top.code.iloc[0], float(top.p.iloc[0])
        sug = ", ".join(f"{a} ({b:.0%})" for a, b in zip(top.code, top.p))
        ocr = top.ocr.iloc[0] > 0
        src = "code lu sur la carte" if ocr else "couleur de la légende"
        if pr >= sure:
            out[p] = dict(code=c, source_nom=src, confiance=round(pr, 2), texte_lu=lu_txt.get((p, c), ""), suggestions=sug)
        elif pr >= check:   # trop peu fiable sur cartes réelles (31-64 % juste) : pas de nom, mais suggestion classée en tête
            out[p] = dict(code="", source_nom="non identifiée – suggestion probable", confiance=round(pr, 2), texte_lu=lu_txt.get((p, c), ""), suggestions=sug)
        else:
            out[p] = dict(code="", source_nom="non identifiée", confiance=round(pr, 2), texte_lu="", suggestions=sug)
    return pd.DataFrame.from_dict(out, orient="index")


def poly_signatures(rgb, lab):
    """Par polygone : aire, centre (y, x) et signature couleur/motif (swatch_features)."""
    labc = sk_rgb2lab(rgb).reshape(-1, 3)
    ids = np.unique(lab[lab > 0]); order = np.argsort(lab.ravel(), kind="stable"); sl = lab.ravel()[order]
    st, en = np.searchsorted(sl, ids), np.searchsorted(sl, ids, side="right")
    F = np.stack([swatch_features(labc[order[a:b]]) for a, b in zip(st, en)])
    cy, cx = np.array(ndi.center_of_mass(np.ones_like(lab), lab, ids)).reshape(-1, 2).T
    return pd.DataFrame(dict(aire=en - st, cy=cy, cx=cx), index=ids), F


PROP_PKL = os.path.join(HERE, "prop_model.pkl")


def propagate_codes(rgb, lab, N, prop_pkl=None, legend=None):
    """Piste 1 : les codes lus avec certitude sont propagés aux polygones de même couleur/motif sur TOUTE la carte."""
    import propagate as PR
    prop_pkl = PROP_PKL if prop_pkl is None else prop_pkl
    if not prop_pkl or not os.path.exists(prop_pkl):
        return N
    S, F = poly_signatures(rgb, lab)
    P = S.join(N)
    P["code"] = P.code.fillna(""); P["source_nom"] = P.source_nom.fillna("non identifiée")
    water = P.source_nom.values == "eau ou hors carte"
    lf = dict(zip(legend.index, legend.feat.values)) if legend is not None and "feat" in legend else None
    Q = PR.propagate(P.assign(source_nom=np.where(water, "eau", P.source_nom)), F, pickle.load(open(prop_pkl, "rb")), legend_feats=lf)
    Q["source_nom"] = np.where(water, "eau ou hors carte", Q.source_nom)
    return Q[N.columns]


def merge_same_code(lab, N):
    """Fusionne les polygones VOISINS portant le même code (un polygone par unité continue)."""
    code = N.code.reindex(np.arange(lab.max() + 1)).fillna("").values
    par = np.arange(lab.max() + 1)
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = a != b
        pr = np.unique(np.stack([a[m], b[m]], 1), axis=0)
        for x, y in pr:
            if code[x] != "" and code[x] == code[y]:
                rx, ry = find(x), find(y)
                if rx != ry:
                    par[max(rx, ry)] = min(rx, ry)
    root = np.array([find(i) for i in range(len(par))])
    if (root == np.arange(len(par))).all():
        return lab, N
    rank = {"code lu sur la carte": 0, "même couleur qu'une unité lue sur la carte": 1}
    M = N.copy(); M["root"] = root[M.index]; M["_r"] = M.source_nom.map(rank).fillna(9); M["_a"] = np.bincount(lab.ravel(), minlength=len(par))[M.index]
    M = M.sort_values(["root", "_r", "_a"], ascending=[True, True, False])
    G = M.groupby("root").first()
    G["texte_lu"] = M.groupby("root").texte_lu.apply(lambda s: ", ".join(dict.fromkeys(t for t in s.fillna("") if t)))
    G["confiance"] = M.groupby("root").confiance.max()
    return root[lab], G.drop(columns=["_r", "_a"])


def absorb_slivers(rgb, lab, N, max_px=200, max_dist=25.0, passes=3):
    """Éclats NON nommés (< max_px) laissés par la fusion des échelles : absorbés par le voisin avec lequel ils partagent la plus
    longue frontière, si ce voisin est nommé et de couleur proche (distance des signatures <= max_dist)."""
    for _ in range(passes):
        S, F = poly_signatures(rgb, lab)
        code = N.code.reindex(S.index).fillna("").values; src = N.source_nom.reindex(S.index).fillna("").values
        small = (S.aire.values < max_px) & (code == "") & (src != "eau ou hors carte")
        if not small.any():
            break
        pos = pd.Series(np.arange(len(S)), index=S.index)
        pairs = []
        for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
            m = (a != b) & (a > 0) & (b > 0)
            pairs.append(np.stack([a[m], b[m]], 1)); pairs.append(np.stack([b[m], a[m]], 1))
        E = pd.DataFrame(np.concatenate(pairs), columns=["a", "b"]).value_counts().rename("n").reset_index()
        ia, ib = pos[E.a].values, pos[E.b].values
        E = E[small[ia] & (code[ib] != "")]
        ia, ib = pos[E.a].values, pos[E.b].values
        E["d"] = np.sqrt(((F[ia] - F[ib]) ** 2).sum(1))
        E = E[E.d <= max_dist].sort_values(["a", "n"], ascending=[True, False]).drop_duplicates("a")
        if not len(E):
            break
        mp = np.arange(lab.max() + 1); mp[E.a.values] = E.b.values
        lab = mp[lab]; N = N.drop(index=E.a.values, errors="ignore")
    return lab, N


WATER_THR = 0.9
LAST = {}   # état intermédiaire du dernier run (évaluation : nommage avant propagation / fusion)

def mark_water(rgb, lab, words, N, water_pkl):
    """Régions sans unité géologique (lac, mer, hors carte) : seulement si AUCUN code sûr n'a été donné par le texte ou la légende
    (ex. la calotte glaciaire 'Ice', blanche comme la mer, reste une unité si la légende la nomme)."""
    if not water_pkl or not os.path.exists(water_pkl):
        return N
    import water as WB
    Wm = pickle.load(open(water_pkl, "rb"))
    F = WB.region_feats(rgb, lab, words).set_index("poly")
    pw = pd.Series(Wm["model"].predict_proba(F[Wm["features"]])[:, 1], index=F.index)
    N = N.copy()
    m = (N.code == "") & (pw.reindex(N.index).fillna(0) >= WATER_THR)
    N.loc[m, "source_nom"] = "eau ou hors carte"; N.loc[m, "confiance"] = pw.reindex(N.index)[m].round(2)
    return N

# ------------------------------------------------------------------ 6. export
QML_UNITS = open(os.path.join(HERE, "qml_units.xml")).read() if os.path.exists(os.path.join(HERE, "qml_units.xml")) else None
QML_FAULTS = open(os.path.join(HERE, "qml_faults.xml")).read() if os.path.exists(os.path.join(HERE, "qml_faults.xml")) else None


def export_gpkg(path, P, F, legend):
    if os.path.exists(path):
        os.remove(path)
    P.to_file(path, layer="unites_geologiques", driver="GPKG")
    # couche failles toujours présente (même vide) : l'utilisateur la retrouve dans QGIS et peut y dessiner
    F.to_file(path, layer="failles", driver="GPKG", geometry_type="LineString")
    con = sqlite3.connect(path)
    legend.drop(columns=["feat"]).reset_index().rename(columns={"index": "code"}).to_sql("legende", con, index=False, if_exists="replace")
    con.execute("""CREATE TABLE IF NOT EXISTS layer_styles (id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT(256),
      f_table_schema TEXT(256), f_table_name TEXT(256), f_geometry_column TEXT(256), styleName TEXT(30), styleQML TEXT,
      styleSLD TEXT, useAsDefault BOOLEAN, description TEXT, owner TEXT(30), ui TEXT(30), update_time DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    for tbl, qml in (("unites_geologiques", QML_UNITS), ("failles", QML_FAULTS)):
        if qml:
            con.execute("INSERT INTO layer_styles (f_table_catalog,f_table_schema,f_table_name,f_geometry_column,styleName,styleQML,styleSLD,useAsDefault,description,owner,update_time) "
                        "VALUES ('','',?,'geom',?,?,'',1,'style GeoRas2Vec','',strftime('%Y-%m-%dT%H:%M:%fZ','now'))", (tbl, tbl, qml))
    con.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier) SELECT 'layer_styles','attributes','layer_styles' "
                "WHERE NOT EXISTS (SELECT 1 FROM gpkg_contents WHERE table_name='layer_styles')")
    con.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier) SELECT 'legende','attributes','legende' "
                "WHERE NOT EXISTS (SELECT 1 FROM gpkg_contents WHERE table_name='legende')")
    con.commit(); con.close()


BLOCK = int(os.environ.get("GEORAS2VEC_BLOCK", 2048))   # grandes cartes : découpage, lecture et nommage par blocs (mémoire)


def analyse_blocks(rgb, model, legend, reader, naming_pkl, ranker_pkl, water_pkl, tta=False, log=print, B=None):
    """Segmentation + OCR + nommage + eau, bloc par bloc ; étiquettes globales (la propagation et la fusion se font ensuite
    sur TOUTE la carte, ce qui recolle les unités coupées par les bords de blocs)."""
    B = B or BLOCK; H, W = rgb.shape[:2]; t = time.time()
    lab = np.zeros((H, W), np.int32); pf = np.zeros((H, W), np.float32); Ns = []; off = 0
    allow = code_alphabet(legend.index) if OCR_ALLOW else None
    for y in range(0, H, B):
        for x in range(0, W, B):
            sub = np.ascontiguousarray(rgb[y:y + B, x:x + B])
            l, p = segment(model, sub, tta=tta)
            words = read_text(sub, reader, allow=allow)
            N = name_polygons(sub, l, words, legend, naming_pkl, flt=(p > 0.5).astype(np.uint8), ranker_pkl=ranker_pkl)
            N = mark_water(sub, l, words, N, water_pkl)
            l = np.where(l > 0, l + off, 0).astype(np.int32); N.index = N.index + off
            lab[y:y + B, x:x + B] = l; pf[y:y + B, x:x + B] = p; Ns.append(N); off = max(off, int(l.max()))
            log(f"bloc ({y}, {x}) : {len(N)} polygones, {len(words)} textes ({time.time() - t:.0f} s)")
    return lab, pf, pd.concat(Ns)


def finish(rgb, lab, pf, N, transform, crs, legend, out_gpkg):
    """Vectorisation + table attributaire + failles + export GeoPackage."""
    P = vectorize_units(lab, transform, crs)
    P = P.join(N, on="pid")
    P["code"] = P.code.fillna(""); P["source_nom"] = P.source_nom.fillna("non identifiée")
    lg = legend.reindex(P.code)
    P["nom"] = np.where(P.code != "", (P.code + " – " + lg.lithologie.fillna("").values).str.strip(" –"), "Unité non identifiée")
    P["nom"] = np.where(P.source_nom == "eau ou hors carte", "Eau, glace ou hors carte (pas d'unité géologique visible)", P.nom)
    P["age"] = lg.age.fillna("").values; P["lithologie"] = lg.lithologie.fillna("").values
    med = [ndi.median(rgb[..., c], lab, P.pid.values) for c in range(3)]
    P["couleur_carte"] = ["#%02x%02x%02x" % tuple(int(v) for v in x) for x in zip(*med)]
    P["couleur_hex"] = np.where(P.code != "", lg.couleur_hex.fillna("").values, P.couleur_carte)
    P["couleur_hex"] = np.where(P.couleur_hex == "", P.couleur_carte, P.couleur_hex)
    P["surface_km2"] = (P.area / 1e6).round(3)
    P["id"] = np.arange(1, len(P) + 1)
    P = P[["id", "code", "nom", "age", "lithologie", "source_nom", "confiance", "texte_lu", "suggestions",
           "couleur_hex", "couleur_carte", "surface_km2", "geometry"]]
    F = vectorize_faults(pf, transform, crs)
    export_gpkg(out_gpkg, P, F, legend.loc[legend.index.isin(P.code)])
    return P, F


def run(rgb, transform, crs, model, legend, reader, naming_pkl, out_gpkg, tta=False, log=print, ranker_pkl="legend_ranker.pkl",
        water_pkl=os.path.join(HERE, "water_model.pkl"), prop_pkl=None):
    t = time.time()
    lab, pf, N = analyse_blocks(rgb, model, legend, reader, naming_pkl, ranker_pkl, water_pkl, tta=tta, log=log)
    LAST.update(lab0=lab, N0=N.copy())
    N = propagate_codes(rgb, lab, N, prop_pkl, legend); log(f"propagation par la couleur ({time.time() - t:.0f} s)")
    LAST.update(N_prop=N.copy())
    lab, N = merge_same_code(lab, N)
    LAST.update(lab1=lab, N1=N.copy())
    P, F = finish(rgb, lab, pf, N, transform, crs, legend, out_gpkg)
    log(f"export : {out_gpkg} | {len(P)} unités, {len(F)} failles ({time.time() - t:.0f} s)")
    return P, F, lab, pf
