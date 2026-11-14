# Test de bout en bout sur une vraie zone GMNA (jamais vue par le nommage : zones nord)
#   python e2e_test.py <modele.pt> <zone naming_cache/north_*.pkl> <sortie_dir> [tta]
import sys, os, glob, pickle, re, time
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
import numpy as np, pandas as pd, rasterio, easyocr
from rasterio.windows import Window
from skimage.color import rgb2lab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import build_tiles as G, pipeline as PL

ckpt, region, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TTA = len(sys.argv) > 4 and sys.argv[4] == "tta"
os.makedirs(OUT, exist_ok=True)
W = os.environ.get("GEORAS2VEC_RAW", "data/world_dataset/")
norm = lambda c: re.sub(r"[\*\?\s]", "", str(c))

# --- légende : une case par code (couleur / motif imprimés, pris sur les zones SUD) + âge, lithologie
if os.path.exists("legend_gmna.pkl"):
    LEG = pd.read_pickle("legend_gmna.pkl")
else:
    with rasterio.open(W + "na_north/gmna_north_300dpi.tif") as s:
        crs = s.crs
    gm = G.load_gmna(W + "na_shapes/GMNA_SHAPES/Geologic_units.shp", W + "na_shapes/GMNA_SHAPES/Faults.shp", crs)
    meta = {norm(r.code): (str(r.age or ""), str(r.lith or "").strip()) for r in gm.tab.itertuples()}
    feats, cols = {}, {}
    for f in sorted(glob.glob("naming_cache/south_*.pkl")):
        r = pickle.load(open(f, "rb")); inst = r["inst"]; lab = rgb2lab(r["rgb"]).reshape(-1, 3); rgbf = r["rgb"].reshape(-1, 3)
        area = np.bincount(inst.ravel())
        order = np.argsort(inst.ravel(), kind="stable"); sl = inst.ravel()[order]
        for i in np.nonzero(area >= 400)[0]:
            if i == 0:
                continue
            a, b = np.searchsorted(sl, i), np.searchsorted(sl, i, side="right")
            c = norm(r["codes"][i - 1])
            feats.setdefault(c, []).append(PL.swatch_features(lab[order[a:b]]))
            cols.setdefault(c, []).append(np.median(rgbf[order[a:b]], 0))
    LEG = pd.DataFrame({c: dict(nom=c, age=meta.get(c, ("", ""))[0].replace(" / ", " – ").strip(" –"),
                                lithologie=meta.get(c, ("", ""))[1],
                                couleur_hex="#%02x%02x%02x" % tuple(int(v) for v in np.median(cols[c], 0)),
                                feat=np.median(np.array(v), 0)) for c, v in feats.items()}).T
    LEG.to_pickle("legend_gmna.pkl")
print("légende :", len(LEG), "cases")

# --- zone test (raster + géoréférencement) ; légende "régionale" = unités présentes dans la zone (comme une vraie carte)
R = pickle.load(open(region, "rb"))
name = os.path.basename(region)[:-4]; part, r0, c0 = name.split("_"); r0, c0 = int(r0), int(c0)
with rasterio.open(W + f"na_{part}/gmna_{part}_300dpi.tif") as s:
    tr = rasterio.windows.transform(Window(c0, r0, 2048, 2048), s.transform); crs = s.crs
rgb = R["rgb"]
true = np.array([""] + [norm(c) for c in R["codes"]])
# légende de la FEUILLE (comme imprimée sur la carte) : toutes les unités de la feuille, cases mesurées hors zone testée
z = pickle.load(open(f"bench_cache/{name}.pkl", "rb"))
local = LEG.reindex(z["legend"].index).copy()
local["feat"] = z["legend"].feat.values
for c in ("nom", "age", "lithologie", "couleur_hex"):
    local[c] = local[c].fillna(local.index.to_series() if c == "nom" else "")
print("légende de la feuille :", len(local), "unités")

model = PL.load_model(ckpt)
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
RANKER = os.environ.get("RANKER", "legend_ranker_sud.pkl")
P, F, lab, pf = PL.run(rgb, tr, crs, model, local, reader, "naming_model.pkl", f"{OUT}/carte_geologique.gpkg", tta=TTA, ranker_pkl=RANKER, water_pkl=os.environ.get("WATER", "water_model_sud.pkl"))

# --- mesure (on connaît la vérité de cette zone)
gt = R["inst"]
pid_code = dict(zip(P.index + 1, P.code)) if False else None
code_of = {}
import geopandas as gpd
N = PL.name_polygons  # (déjà fait dans run) ; on relit les codes par polygone du raster
code_px = np.full(lab.max() + 1, "", dtype=object)
# P a été construit dans l'ordre des pid ; on retrouve pid -> code via un rasterisé des centroïdes : plus simple, on refait la jointure
Pp = PL.vectorize_units(lab, tr, crs)[["pid"]]
assert len(Pp) == len(P)
for pid, c in zip(Pp.pid, P.code):
    code_px[pid] = c
pred_code = code_px[lab]
true_code = true[gt]
valid = gt > 0
named = valid & (pred_code != "")
print(f"\n=== RÉSULTAT BOUT EN BOUT ({name}, TTA={TTA}) ===")
print(f"surface nommée : {named.sum() / valid.sum():.1%} | nom juste sur la surface nommée : {(pred_code == true_code)[named].mean():.1%} "
      f"| nom juste sur toute la surface : {(pred_code == true_code)[valid].mean():.1%}")
src = np.full(lab.max() + 1, "", dtype=object)
for pid, s_ in zip(Pp.pid, P.source_nom):
    src[pid] = s_
S = src[lab]
for s_ in sorted(set(S[valid])):
    m = valid & (S == s_)
    print(f"  {s_:<24} : {m.sum() / valid.sum():6.1%} de la surface, nom juste {((pred_code == true_code)[m].mean() if m.any() else 0):6.1%}")
Pc = P.assign(vrai=[true_code[lab == pid].tolist() and pd.Series(true_code[lab == pid]).mode()[0] for pid in Pp.pid])
for lo, hi in ((0.8, 1.01), (0.5, 0.8)):
    m = (Pc.code != "") & (Pc.confiance >= lo) & (Pc.confiance < hi)
    print(f"  confiance [{lo}, {hi}) : {m.sum()} polygones, nom juste {(Pc.code == Pc.vrai)[m].mean() if m.any() else 0:.1%}")
Pc["aire_px"] = [int((lab == pid).sum()) for pid in Pp.pid]
Pc["eau_vrai"] = [float((gt[lab == pid] == 0).mean()) for pid in Pp.pid]
Pc["top1"] = Pc.suggestions.fillna("").str.split(" \\(").str[0]
Pc["p1"] = pd.to_numeric(Pc.suggestions.fillna("").str.extract(r"\((\d+)%\)")[0], errors="coerce") / 100
Pc.drop(columns="geometry").to_csv(f"{OUT}/polygones_verite.csv", index=False)
P.drop(columns="geometry").to_csv(f"{OUT}/table_attributaire.csv", index=False)
# --- eau / hors carte : trous de la référence (inst == 0) détectés ?
hole = gt == 0; wat = S == "eau ou hors carte"
print(f"  eau/hors carte : {hole.mean():.1%} de la zone ; détectée {(wat & hole).sum() / max(hole.sum(), 1):.1%} ; unités prises pour de l'eau {(wat & valid).sum() / valid.sum():.2%}")
# --- ligne de résultat (agrégation multi-zones)
import csv
ok = (pred_code == true_code)
row = dict(zone=name, modele=os.path.basename(os.path.dirname(os.path.abspath(ckpt))) + "/" + os.path.basename(ckpt), surface=int(valid.sum()),
           nommee=int(named.sum()), juste=int(ok[named].sum()),
           sure=int((valid & np.isin(S, ["code lu sur la carte", "couleur de la légende"])).sum()),
           sure_juste=int((valid & np.isin(S, ["code lu sur la carte", "couleur de la légende"]) & ok).sum()),
           trous=int(hole.sum()), eau_ok=int((wat & hole).sum()), eau_faux=int((wat & valid).sum()))
fn = os.environ.get("E2E_CSV", "e2e_resultats.csv"); new = not os.path.exists(fn)
with open(fn, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
