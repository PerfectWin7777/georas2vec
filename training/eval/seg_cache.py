# Cache des sorties du modèle (affinités + failles, uint8) sur validation (échantillon par source) et test complet.
#   python seg_cache.py <modele.pt> <dossier_parquet> <sortie>
import sys, os, io, glob, numpy as np, pyarrow.parquet as pq, torch
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL
torch.set_num_threads(4)
ckpt, data, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
model = PL.load_model(ckpt)
N_VAL = 30
for split in ("validation", "test"):
    rows = []
    for f in sorted(glob.glob(f"{data}/{split}-*.parquet")):
        rows += pq.read_table(f, columns=["id", "source", "kind", "image", "instances", "faults"]).to_pylist()
    if split == "validation":
        keep = []
        for key in sorted({(r["source"], r["kind"]) for r in rows}):
            g = [r for r in rows if (r["source"], r["kind"]) == key]
            keep += [g[i] for i in np.linspace(0, len(g) - 1, min(N_VAL, len(g))).astype(int)]
        rows = keep
    for r in rows:
        fn = f"{out}/{split}__{r['id']}.npz"
        if os.path.exists(fn):
            continue
        rgb = np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
        inst = np.asarray(Image.open(io.BytesIO(r["instances"]["bytes"]))).astype(np.int32)
        flt = (np.asarray(Image.open(io.BytesIO(r["faults"]["bytes"]))) > 0)
        aff, pf = PL.infer(model, rgb)
        np.savez_compressed(fn, aff=(aff * 255).round().astype(np.uint8), pf=(pf * 255).round().astype(np.uint8),
                            inst=inst.astype(np.uint16), flt=flt, source=r["source"], kind=r["kind"])
    print(split, len(rows), flush=True)
