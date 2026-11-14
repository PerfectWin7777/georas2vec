# Fusion des deux échelles : intersection des régions (échelle 1 sépare mieux les grandes unités, 1.25 trouve les petites),
# puis les éclats < min px sont rattachés au voisin. Validation (30 réelles + 30 synthétiques) + zones propres.
import sys, os, io, glob, pickle, numpy as np, pandas as pd, pyarrow.parquet as pq, torch
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [os.path.join(_ROOT, "georas2vec"), os.path.join(_ROOT, "dataset"), os.path.join(_ROOT, "training", "eval")]
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL, seg_eval as SE
torch.set_num_threads(3)
model = PL.load_model(os.environ.get("GEORAS2VEC_WEIGHTS", "best_fp16.pt"))


def fuse(a, b, mn):
    _, inv = np.unique(a.astype(np.int64) * (b.max() + 1) + b, return_inverse=True)
    return SE.remove_small(inv.reshape(a.shape) + 1, mn)


def variants(rgb, inst, flt):
    a, pf = PL.segment(model, rgb, scale=1.0); b, _ = PL.segment(model, rgb, scale=1.25)
    out = {"échelle 1": a, "échelle 1.25": b}
    for mn in (20, 40, 80):
        out[f"fusion min {mn}"] = fuse(a, b, mn)
    return {k: SE.metrics(v, pf, inst, flt, 0.2) for k, v in out.items()}


rows = []
for z in ("north_0_4096", "north_4096_18432"):
    R = pickle.load(open(f"naming_cache/{z}.pkl", "rb"))
    for k, m in variants(R["rgb"], R["inst"].astype(np.int32), (R["flt"] > 0).astype(np.uint8)).items():
        rows.append(dict(kind="zone " + z, variante=k, **m))
    print(z, flush=True)
cached = {os.path.basename(f).split("__", 1)[1][:-4]: f for f in glob.glob("../seg_cache/validation__*.npz")}
T = []
for f in sorted(glob.glob("data/georas2vec/validation-*.parquet")):
    T += [r for r in pq.read_table(f, columns=["id", "kind", "image"]).to_pylist() if r["id"] in cached]
real = [r for r in T if r["kind"] == "real"][:30]; syn = [r for r in T if r["kind"] != "real"]
syn = [syn[i] for i in np.linspace(0, len(syn) - 1, 30).astype(int)]
for r in real + syn:
    z = np.load(cached[r["id"]]); rgb = np.asarray(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
    for k, m in variants(rgb, z["inst"].astype(np.int32), z["flt"]).items():
        rows.append(dict(kind=r["kind"], variante=k, **m))
D = pd.DataFrame(rows); D.to_pickle("fuse_bench.pkl")
for k, g in D.groupby("kind"):
    print(k); print(g.groupby("variante").apply(SE.summarize).round(4).to_string())
