"""Eau / glace / hors carte : indices par région (couleur, texture, texte non géologique). Aucun effet de bord à l'import."""
import re, numpy as np, pandas as pd
from skimage.color import rgb2lab
import pipeline as PL, naming_model as NM

WFEATS = ["L", "a", "b", "Lc", "ac", "bc", "Lf", "af", "bf", "sL", "sa", "sb", "log_aire", "blanc", "bleu", "chroma",
          "bord", "n_mots", "n_mots_codes", "compacite"]


def region_feats(rgb, lab, words=None, codes_vocab=None):
    """Indices par région (lab = étiquettes 1..N) pour reconnaître l'eau / le hors-carte."""
    L = rgb2lab(rgb); Lf = L.reshape(-1, 3)
    ids = np.unique(lab[lab > 0])
    order = np.argsort(lab.ravel(), kind="stable"); sl = lab.ravel()[order]
    st = np.searchsorted(sl, ids); en = np.searchsorted(sl, ids, side="right")
    rows = []
    border = np.zeros(lab.max() + 1, bool)
    for e in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        border[e] = True
    per = np.zeros(lab.max() + 1)
    b = np.zeros(lab.shape, bool); b[:, 1:] |= lab[:, 1:] != lab[:, :-1]; b[1:] |= lab[1:] != lab[:-1]
    np.add.at(per, lab[b], 1)
    nw = np.zeros(lab.max() + 1); nwc = np.zeros(lab.max() + 1)
    if words is not None and len(words):
        H, W = lab.shape
        for w in words.itertuples():
            i = lab[int(np.clip(w.cy, 0, H - 1)), int(np.clip(w.cx, 0, W - 1))]
            if len(re.sub("[^A-Za-z]", "", w.text)) >= 3:
                nw[i] += 1
                if NM.to_codes(w.text):
                    nwc[i] += 1
    for i, a, e in zip(ids, st, en):
        px = Lf[order[a:e]]
        f = PL.swatch_features(px)
        ch = np.hypot(px[:, 1], px[:, 2])
        rows.append(dict(poly=i, **dict(zip(WFEATS[:12], f)), log_aire=np.log(e - a),
                         blanc=float(((px[:, 0] > 90) & (ch < 6)).mean()), bleu=float(((px[:, 2] < -8) & (px[:, 0] > 55)).mean()),
                         chroma=float(np.median(ch)), bord=float(border[i]), n_mots=nw[i], n_mots_codes=nwc[i],
                         compacite=float(4 * np.pi * (e - a) / max(per[i], 1) ** 2)))
    return pd.DataFrame(rows)


