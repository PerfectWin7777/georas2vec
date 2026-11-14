# Inférence sur image agrandie (multi-échelle) pour les petits polygones : modèle sur image x s, MWS, retour à la résolution
# d'origine (plus proche voisin), mêmes métriques que seg_eval. Validation : 30 tuiles réelles + 30 synthétiques.
import sys, os, io, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, torch
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
from PIL import Image
from skimage.transform import resize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL, seg_eval as SE
torch.set_num_threads(4)
SCALES = [float(x) for x in os.environ.get("SCALES", "1.5").split(",")]
SPLIT = os.environ.get("SPLIT", "validation"); NR = int(os.environ.get("NREAL", 30)); NS = int(os.environ.get("NSYN", 30))
model = PL.load_model(os.environ.get("GEORAS2VEC_WEIGHTS", "best_fp16.pt"))
cached = {os.path.basename(f).split("__", 1)[1][:-4]: f for f in glob.glob(f"../seg_cache/{SPLIT}__*.npz")}
rows = []
for f in sorted(glob.glob(f"data/georas2vec/{SPLIT}-*.parquet")):
    rows += [r for r in pq.read_table(f, columns=["id", "kind", "image"]).to_pylist() if r["id"] in cached]
real = [r for r in rows if r["kind"] == "real"][:NR]; syn = [r for r in rows if r["kind"] != "real"]
syn = [syn[i] for i in np.linspace(0, len(syn) - 1, NS).astype(int)]
out = []
for r in real + syn:
    z = np.load(cached[r["id"]]); inst = z["inst"].astype(np.int32); flt = z["flt"]; pf0 = z["pf"] / 255
    rgb = np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
    base = SE.mws(z["aff"].astype(np.float32) / 255, min_size=20)
    out.append(dict(variante="échelle 1", kind=r["kind"], **SE.metrics(base, pf0, inst, flt)))
    for s in SCALES:
        big = (resize(rgb, (int(512 * s), int(512 * s)), order=1, anti_aliasing=False) * 255).astype(np.uint8)
        aff, pf = PL.infer(model, big)
        lab = SE.mws(aff, min_size=int(20 * s * s))
        lab = lab[(np.arange(512) * s + s / 2).astype(int)][:, (np.arange(512) * s + s / 2).astype(int)]
        lab = SE.remove_small(lab, 20)
        out.append(dict(variante=f"échelle {s}", kind=r["kind"], **SE.metrics(lab, pf0, inst, flt)))
    print(len(out), flush=True)
D = pd.DataFrame(out); D.to_pickle(os.environ.get("OUTP", "scale_bench.pkl"))
for k, g in D.groupby("kind"):
    print(k); print(g.groupby("variante").apply(SE.summarize).round(4).to_string())
