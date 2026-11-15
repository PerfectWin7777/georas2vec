# Test officiel USGS (78 tuiles réelles) avec la chaîne publiée (fusion 2 échelles + failles vectorisées), deux jeux de poids.
import os, io, glob, sys, numpy as np, pandas as pd, pyarrow.parquet as pq, torch
from PIL import Image
from rasterio.transform import Affine
sys.path.insert(0, "."); import pipeline as PL, seg_eval as SE
from fault_bench import f1
torch.set_num_threads(4)
rows = []
for f in sorted(glob.glob("../hf_ds_v2/data/test-*.parquet")):
    rows += [r for r in pq.read_table(f, columns=["id", "kind", "image", "instances", "faults"]).to_pylist() if r["kind"] == "real"]
out = []
for nm, w in (("actuel", "../train_out2/georas2vec_model/best_fp16.pt"), ("complet", "../full_out/georas2vec_model/best_fp16.pt")):
    model = PL.load_model(w); TP = FP = FN = 0
    for r in rows:
        rgb = np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
        inst = np.asarray(Image.open(io.BytesIO(r["instances"]["bytes"]))).astype(np.int32)
        flt = (np.asarray(Image.open(io.BytesIO(r["faults"]["bytes"]))) > 0).astype(np.uint8)
        lab, pf = PL.segment(model, rgb); m = SE.metrics(lab, pf, inst, flt, 0.2)
        F = PL.vectorize_faults(pf, Affine.identity(), None)
        a, b, c = f1([[(int(y), int(x)) for x, y in np.array(g.coords)] for g in F.geometry], flt > 0); TP += a; FP += b; FN += c
        out.append(dict(poids=nm, **m))
    p, rc = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    D = pd.DataFrame([o for o in out if o["poids"] == nm])
    print(f"{nm:<8} grandes {D.b_ok.sum()/D.b_n.sum():.1%} petites {D.s_ok.sum()/D.s_n.sum():.1%} surface {D.surf_ok.sum()/D.surf_n.sum():.1%} failles(lignes) F1 {2*p*rc/max(p+rc,1e-9):.3f}", flush=True)
