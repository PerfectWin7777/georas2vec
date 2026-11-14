import os as _o, sys as _s
_R = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..") if "eval" in __file__ else _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..")
_s.path[:0] = [_o.path.join(_R, "georas2vec"), _o.path.join(_R, "training", "eval"), _o.path.join(_R, "training")]
# Données du classeur de nommage avec la VRAIE légende imprimée (lue automatiquement puis vérifiée) et le lecteur de codes.
import os, sys, glob, pickle, re, difflib, numpy as np, pandas as pd
from scipy import ndimage as ndi
sys.path.insert(0, "."); import pipeline as PL, code_reader as CR
from naming_v08_test import auto_legend, canon
V = pickle.load(open("code_vocab.pkl", "rb"))["gmna"]
if not os.path.exists("legend_auto_verified.pkl"):
    Lc = auto_legend("crnn").copy()
    snap = {c: (c if c in V else (difflib.get_close_matches(c, V, n=1, cutoff=0.5) or [c])[0]) for c in Lc.index}
    Ld = Lc.copy(); Ld.index = [snap[c] for c in Lc.index]; Ld = Ld[~Ld.index.duplicated()]; Ld.to_pickle("legend_auto_verified.pkl")
Ld = pd.read_pickle("legend_auto_verified.pkl"); codes = np.array(Ld.index); Fm = np.stack(Ld.feat.values)
RD = CR.Reader("code_reader.pt"); os.makedirs("bench_auto", exist_ok=True); parts = []
for f in sorted(glob.glob("naming_cache/north_*.pkl") + glob.glob("naming_cache/south_*.pkl")):
    name = os.path.basename(f)[:-4]; out = f"bench_auto/{name}.pkl"
    if os.path.exists(out):
        parts.append(pd.read_pickle(out)); continue
    R = pickle.load(open(f, "rb")); z = pickle.load(open(f"bench_cache3/{name}.pkl", "rb"))
    P = z["polys"]; P = P[P.area >= 40].reset_index(drop=True)
    words = PL.word_ink(R["rgb"], PL.reread_codes(R["rgb"], R["words"], RD))
    D, ch = PL.ocr_candidates(R["rgb"], R["inst"], words, codes, "naming_model.pkl", (R["flt"] > 0).astype(np.uint8))
    cents = ndi.center_of_mass(np.ones_like(R["inst"]), R["inst"], P.poly.values)
    X = PL.rank_pairs(P.poly.values, np.stack(P.feat.values), P.area.values, cents, codes, Fm, D, ch, words, lab=R["inst"])
    tr = dict(zip(P.poly, P.code.map(canon))); ar = dict(zip(P.poly, P.area))
    X["zone"] = name; X["vrai"] = X.poly.map(tr); X["aire"] = X.poly.map(ar); X["y"] = (X.code.map(canon) == X.vrai).astype(int)
    X.to_pickle(out); parts.append(X); print(name, len(X), flush=True)
pd.concat(parts, ignore_index=True).to_pickle("rank_auto.pkl"); print("fini", flush=True)
