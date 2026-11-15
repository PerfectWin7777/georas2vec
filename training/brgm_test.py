# Test sur une AUTRE série : cartes géologiques du BRGM au 1/50 000 (scans servis par WMS), vérité = BD Charm-50 (Calvados).
# Jamais vue à l'entraînement (ni le style français, ni le fond topographique, ni les codes).
import os, sys, glob, re, numpy as np, pandas as pd, geopandas as gpd, rasterio, torch
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, "."); import pipeline as PL, seg_eval as SE
from fault_bench import f1
from rasterio.transform import Affine
torch.set_num_threads(4)
B = "../brgm"
P = gpd.read_file(f"{B}/dl14/GEO050K_HARM_014_S_FGEOL_2154.shp")
L = gpd.read_file(f"{B}/dl14/GEO050K_HARM_014_L_STRUCT_2154.shp"); L = L[L.DESCR.str.contains("aille", na=False)]
model = PL.load_model(os.environ.get("WEIGHTS", "../train_out2/georas2vec_model/best_fp16.pt"))
rows = []
for f in sorted(glob.glob(f"{B}/img/scan_*.png")):
    x0, y0 = (int(v) for v in re.findall(r"scan_(\d+)_(\d+)", f)[0]); S = 8600; H = W = 2048
    tr = from_bounds(x0, y0, x0 + S, y0 + S, W, H)
    rgb = np.asarray(Image.open(f).convert("RGB"))
    # vérité : une instance = composante connexe d'une même notation
    Q = P.cx[x0:x0 + S, y0:y0 + S]
    codes = {c: k + 1 for k, c in enumerate(sorted(Q.NOTATION.fillna("?").unique()))}
    unit = rasterize(zip(Q.geometry, Q.NOTATION.fillna("?").map(codes)), out_shape=(H, W), transform=tr, fill=0, dtype="int32")
    inst = np.zeros_like(unit); n = 0
    for c in np.unique(unit[unit > 0]):
        lab_c, k = ndi.label(unit == c); inst[lab_c > 0] = lab_c[lab_c > 0] + n; n += k
    # unité minimale de cartographie comme le jeu de données (< 40 px fusionnés au voisin)
    inst = SE.remove_small(inst, 40) if inst.max() else inst
    flt = rasterize(((g, 1) for g in L.cx[x0:x0 + S, y0:y0 + S].geometry), out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
    lab, pf = PL.segment(model, rgb)
    m = SE.metrics(lab, pf, inst, flt, 0.2)
    F = PL.vectorize_faults(pf, Affine.identity(), None)
    tp, fp, fn = f1([[(int(y), int(x)) for x, y in np.array(g.coords)] for g in F.geometry], flt > 0)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    r = dict(zone=os.path.basename(f), unites=int(inst.max()), grandes=m["b_ok"] / max(m["b_n"], 1), petites=m["s_ok"] / max(m["s_n"], 1),
             surface=m["surf_ok"] / max(m["surf_n"], 1), failles_F1=2 * pr * rc / max(pr + rc, 1e-9), failles_vraies=int(flt.sum()))
    rows.append(r); print(r, flush=True)
    np.savez_compressed(f"{B}/img/{os.path.basename(f)[:-4]}_res{os.environ.get("TAG", "")}.npz", lab=lab, pf=(pf * 255).astype(np.uint8), inst=inst, flt=flt)
D = pd.DataFrame(rows); print(D.round(3).to_string()); print(D.mean(numeric_only=True).round(3).to_string())
