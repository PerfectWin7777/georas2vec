# Jeu « couches » : tuile imprimée -> plaque couleur propre + encre étiquetée (+ fond topo sur ~50 %), instances, failles.
#   python build_layers.py N_PAR_SOURCE  -> ../layers_ds/<source>.parquet
import os, io, sys, pickle, random, numpy as np, pandas as pd, multiprocessing as mp
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
import build_tiles as G, topo_aug as TA
OUT = "../layers_ds"; S = {}
# srcs*.pkl : dictionnaires {nom: source} chargés une fois avec build_tiles.load_* (sources vectorielles du jeu de données)
for f in ("srcs.pkl", "srcs2.pkl"):
    S.update(pickle.load(open(f, "rb")))
BS = {k: G.BlockSplit(v, 300e3, 1) for k, v in S.items()}


def png(a, mode=None):
    b = io.BytesIO(); Image.fromarray(a, mode).save(b, "PNG"); return b.getvalue()


def job(args):
    name, split, i = args
    rng = np.random.default_rng(G._seed(77, name, split, i))
    for _ in range(20):
        r = G.synth_tile(S[name], BS[name], split, rng, layers=True)
        if r is None:
            continue
        rgb, inst, Uf, flt, meta = r; ed = meta["edition"]; ink = ed["_ink"].copy(); plate = ed["_plate"]
        if rng.random() < 0.5:
            rgb2 = TA.topo_overlay(rgb, random.Random(int(rng.integers(1 << 30))))
            ink[(np.abs(rgb2.astype(int) - rgb).sum(-1) > 24) & (ink == 0)] = 6; rgb = rgb2
        return dict(id=f"{name}_{split}_{i:05d}", source=S[name].name, split=split, image=png(rgb), plate=png(plate),
                    ink=png(ink), instances=png(inst.astype(np.uint16), "I;16"), faults=png((flt > 0).astype(np.uint8) * 255))
    return None


if __name__ == "__main__":
    # reprise possible : sauvegarde par paquets de 50 tuiles (<source>_<k>.parquet), paquets existants sautés
    n = int(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
    for name in S:
        if os.path.exists(f"{OUT}/{name}.parquet"):
            continue
        for k in range(0, n, 50):
            out = f"{OUT}/{name}_{k:04d}.parquet"
            if os.path.exists(out):
                continue
            jobs = [(name, "train" if i % 10 else "validation", i) for i in range(k, min(n, k + 50))]
            with mp.get_context("fork").Pool(max(1, os.cpu_count() - 1)) as p:
                rows = [r for r in p.imap_unordered(job, jobs, chunksize=2) if r]
            pd.DataFrame(rows).to_parquet(out + ".tmp", index=False); os.replace(out + ".tmp", out); print(name, k, len(rows), flush=True)
    print("FINI", flush=True)
