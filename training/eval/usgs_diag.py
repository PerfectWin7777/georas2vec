# Test officiel USGS (78 tuiles réelles) avec la chaîne publiée (fusion 2 échelles + failles vectorisées), deux jeux de poids.
import os, io, glob, sys, numpy as np, pandas as pd, pyarrow.parquet as pq, torch
from PIL import Image
from rasterio.transform import Affine
sys.path.insert(0, "."); import pipeline as PL, seg_eval as SE
from fault_bench import f1
torch.set_num_threads(4)
rows = []
XM = os.environ.get("GC_XFORM", "")
def XF(rgb, rid):
    """Diagnostic : 'erase' = encre sombre peu colorée effacée (traits noirs, texte) ; 'topo' = fond topographique ajouté."""
    if XM == "erase":
        import cv2
        from skimage.color import rgb2lab
        L = rgb2lab(rgb); ink = (L[..., 0] < 45) & (np.hypot(L[..., 1], L[..., 2]) < 18)
        ink = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8))
        return cv2.inpaint(np.ascontiguousarray(rgb), ink, 5, cv2.INPAINT_TELEA)
    if XM in ("clean", "topo_clean"):
        global SEPM
        import random, topo_aug as TA, separator as SP
        if "SEPM" not in globals():
            SEPM = SP.Separator(pretrained=False); SEPM.load_state_dict(torch.load("separator.pt", map_location="cpu"))
        if XM == "topo_clean":
            rgb = TA.topo_overlay(np.ascontiguousarray(rgb), random.Random(hash(rid) % (1 << 30)), strength=0.8)
        pl, pr = SP.separate(SEPM, rgb)
        return SP.clean(rgb, pl, pr, remove=tuple(int(c) for c in os.environ.get("SEP_REMOVE", "6").split(",")))
    if XM == "topo":
        import random, topo_aug as TA
        return TA.topo_overlay(np.ascontiguousarray(rgb), random.Random(hash(rid) % (1 << 30)), strength=0.8)
    return rgb
for f in sorted(glob.glob("../hf_ds_v2/data/test-*.parquet")):
    rows += [r for r in pq.read_table(f, columns=["id", "kind", "image", "instances", "faults"]).to_pylist() if r["kind"] == "real"]
out = []
for nm, w in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
    model = PL.load_model(w); TP = FP = FN = 0
    for r in rows:
        rgb = np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
        rgb = XF(rgb, r["id"])
        inst = np.asarray(Image.open(io.BytesIO(r["instances"]["bytes"]))).astype(np.int32)
        flt = (np.asarray(Image.open(io.BytesIO(r["faults"]["bytes"]))) > 0).astype(np.uint8)
        lab, pf = PL.segment(model, rgb); m = SE.metrics(lab, pf, inst, flt, 0.2)
        F = PL.vectorize_faults(pf, Affine.identity(), None)
        a, b, c = f1([[(int(y), int(x)) for x, y in np.array(g.coords)] for g in F.geometry], flt > 0); TP += a; FP += b; FN += c
        out.append(dict(poids=nm, **m))
    p, rc = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    D = pd.DataFrame([o for o in out if o["poids"] == nm])
    print(f"{nm:<8} grandes {D.b_ok.sum()/D.b_n.sum():.1%} petites {D.s_ok.sum()/D.s_n.sum():.1%} surface {D.surf_ok.sum()/D.surf_n.sum():.1%} failles(lignes) F1 {2*p*rc/max(p+rc,1e-9):.3f}", flush=True)
