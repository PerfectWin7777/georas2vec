"""Lecteur de codes géologiques (CRNN + CTC), entraîné sur des images de codes générées.

Sortie = code CANONIQUE de la base (ex. image « Є D » -> 'CAD', « Ʀm » -> 'TRm', « X³g » -> 'X3g', « ʟKg » -> 'lKg').
Le générateur imite les conventions d'impression : Є (Cambrien), Ʀ (Trias), ℙ (Pennsylvanien), petites majuscules
pour l'âge (ʟ, ᴇ, ᴍ), exposants, astérisque (unité sous-marine), sur cases colorées à motifs ou fonds de carte.
"""
import os, re, random, math, io
import numpy as np, torch, torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONTS = [f for f in ["/usr/share/fonts/truetype/freefont/FreeSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
                     "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"] if os.path.exists(f)]
CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789*?")
IDX = {c: i + 1 for i, c in enumerate(CHARS)}           # 0 = blanc CTC
H, WMAX = 32, 192
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
UNSUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def canon(code):
    """Code de la base -> cible du lecteur (exposants -> chiffres, espaces retirés)."""
    return str(code).translate(UNSUP).replace(" ", "")


def printed(code, rng):
    """Forme imprimée d'un code canonique (aléatoire parmi les conventions observées)."""
    s = code
    if rng.random() < 0.9:
        s = s.replace("CA", "Є")
    if rng.random() < 0.9:
        s = s.replace("TR", "Ʀ")
    if rng.random() < 0.5:
        s = s.replace("IP", "ℙ")
    if rng.random() < 0.5:
        s = re.sub(r"(?<=[A-Z])Z(?=[A-Z]|$)", "z", s.replace("PZ", "Pz").replace("MZ", "Mz"))
    # exposant après X, Y, Z, W (Protérozoïque)
    s = re.sub(r"(?<=[WXYZ])(\d)", lambda m: m.group(1).translate(SUP) if rng.random() < 0.85 else m.group(1), s)
    # préfixes d'âge en petites capitales sur la carte (minuscules dans la légende)
    if rng.random() < 0.4:
        s = re.sub(r"^(\d?)l(?=[A-ZЄƦℙ])", lambda m: m.group(1) + "ʟ", s)
        s = re.sub(r"^(\d?)e(?=[A-ZЄƦℙ])", lambda m: m.group(1) + "ᴇ", s)
    return s


def random_code(rng):
    pre = rng.choice(["", "", "", "l", "u", "m", "e", "pl", "o", "n", "pg", "1", "2", "?"])
    era = rng.choice(["Q", "T", "TQ", "K", "J", "TR", "P", "IP", "M", "D", "S", "O", "CA", "Z", "Y", "X", "W", "A", "PZ", "MZ", "KT", "JK"])
    if era in "XYW" and rng.random() < 0.6:
        era += str(rng.randint(1, 3))
    suf = "".join(rng.choice(["v", "s", "g", "m", "f", "i", "k", "b", "n", "q", "x", "t", "y", "a", "c"]) for _ in range(rng.randint(0, 4)))
    return pre + era + suf + ("*" if rng.random() < 0.25 else "")


def tokens(code, rng):
    """Code canonique -> suite de (texte, échelle, décalage vertical, chevauchement) imitant l'impression des cartes USGS."""
    out, i = [], 0
    smallcap = rng.random() < 0.35
    while i < len(code):
        c2 = code[i:i + 2]
        if c2 == "CA" and rng.random() < 0.9:
            out.append((rng.choice(["Є", "€", "€", "Ɛ"]), 1.0, 0.0, 0.0)); i += 2; continue   # Cambrien : glyphe proche de €
        if c2 == "TR" and rng.random() < 0.9:
            if rng.random() < 0.6:
                out += [("T", 1.0, 0.0, 0.0), ("R", 0.9, 0.05, rng.uniform(0.35, 0.6))]     # T et R superposés
            else:
                out.append(("Ʀ", 1.0, 0.0, 0.0))
            i += 2; continue
        if c2 in ("PZ", "MZ") and rng.random() < 0.85:
            out += [(c2[0], 1.0, 0.0, 0.0), ("z", rng.uniform(0.6, 0.8), rng.uniform(0.0, 0.15), rng.uniform(0.1, 0.35))]
            i += 2; continue
        if c2 == "IP" and rng.random() < 0.4:
            out.append(("ℙ", 1.0, 0.0, 0.0)); i += 2; continue
        ch = code[i]
        if ch.isdigit() and i > 0 and code[i - 1] in "WXYZ" and rng.random() < 0.9:
            out.append((ch, rng.uniform(0.5, 0.7), -rng.uniform(0.25, 0.45), rng.uniform(0.0, 0.1)))     # exposant
        elif ch in "lem" and i <= 1 and i + 1 < len(code) and (code[i + 1].isupper() or code[i + 1] in "lm") and smallcap:
            out.append((ch.upper(), rng.uniform(0.7, 0.8), 0.0, 0.0))                                    # petite capitale
        else:
            out.append((ch, 1.0, 0.0, 0.0))
        i += 1
    return out


def render(code, rng):
    fpath = rng.choice(FONTS); size = rng.randint(16, 34)
    toks = tokens(code, rng)
    fonts = {}
    def F(sc):
        k = max(8, int(size * sc)); fonts.setdefault(k, ImageFont.truetype(fpath, k)); return fonts[k]
    # mise en page des jetons
    pos, x = [], 0.0
    asc = F(1.0).getbbox("H")[1]
    for t, sc, dy, ov in toks:
        f = F(sc); l, tt, r, b = f.getbbox(t); w = r - l
        prev_w = pos[-1][4] if pos else 0
        x -= ov * prev_w
        y = dy * size + (1 - sc) * size * 0.75 if dy >= 0 else dy * size
        pos.append((t, f, x - l, y, w)); x += w + size * 0.04
    tw = int(x) + 2; th = int(size * 1.25)
    bg = tuple(int(v) for v in np.clip(np.array([rng.randint(150, 255) for _ in range(3)]) + rng.randint(-20, 20), 0, 255))
    W, Hh = tw + rng.randint(6, 40), th + rng.randint(8, 22)
    im = Image.new("RGB", (W, Hh), bg); d = ImageDraw.Draw(im)
    for _ in range(rng.randint(0, 60)):
        c = tuple(rng.randint(0, 255) for _ in range(3)); px, py = rng.randint(0, W), rng.randint(0, Hh); s = rng.randint(1, 4)
        k = rng.random()
        if k < 0.45:
            d.ellipse((px, py, px + s, py + s), fill=c)
        elif k < 0.75:
            d.polygon([(px, py), (px + 2 * s, py), (px + s, py - 2 * s)], outline=c)
        elif k < 0.9:
            d.line((px, py, px + rng.randint(-15, 15), py + rng.randint(-15, 15)), fill=c, width=1)
        else:
            d.ellipse((px, py, px + 4 * s, py + 2 * s), fill=c)                                            # taches
    if rng.random() < 0.5:
        d.rectangle((0, 0, W - 1, Hh - 1), outline=(30, 30, 30), width=rng.randint(1, 2))
    ink = tuple([rng.randint(0, 60)] * 3) if rng.random() < 0.9 else tuple(rng.randint(0, 120) for _ in range(3))
    ox = (W - tw) // 2 + rng.randint(-3, 3); oy = (Hh - th) // 2 + rng.randint(-2, 2) + int(size * 0.15)
    for t, f, px, py, w in pos:
        d.text((ox + px, oy + py - asc * 0.2), t, font=f, fill=ink)
    if rng.random() < 0.3:
        im = im.rotate(rng.uniform(-3, 3), expand=False, fillcolor=bg)
    if rng.random() < 0.6:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))
    a = np.asarray(im).astype(np.float32)
    a += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0, 8), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        bio = io.BytesIO(); im.save(bio, "JPEG", quality=rng.randint(40, 90)); im = Image.open(bio).convert("RGB")
    return im


def to_tensor(im):
    """Hauteur 32, largeur <= WMAX (remplissage blanc à droite)."""
    w = max(8, min(WMAX, int(round(im.width * H / im.height))))
    a = np.asarray(im.convert("RGB").resize((w, H), Image.BILINEAR)).astype(np.float32) / 255.0
    out = np.ones((H, WMAX, 3), np.float32); out[:, :w] = a
    return torch.from_numpy(out.transpose(2, 0, 1)), w


class CRNN(nn.Module):
    def __init__(self, n=len(CHARS) + 1):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.cnn = nn.Sequential(c(3, 32), nn.MaxPool2d(2), c(32, 64), nn.MaxPool2d(2), c(64, 128), c(128, 128), nn.MaxPool2d((2, 1)),
                                 c(128, 192), c(192, 192), nn.MaxPool2d((2, 1)), c(192, 256), nn.MaxPool2d((2, 1)))   # H 32 -> 1, W/4
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(256, n)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)             # B, T, C
        return self.fc(self.rnn(f)[0])                          # B, T, n


def decode(logits):
    out = []
    for seq in logits.argmax(-1).cpu().numpy():
        s, prev = [], 0
        for k in seq:
            if k != prev and k != 0:
                s.append(CHARS[k - 1])
            prev = k
        out.append("".join(s))
    return out


class Reader:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.m = CRNN(); self.m.load_state_dict(ck); self.m.eval()

    @torch.no_grad()
    def read(self, images):
        xs = torch.stack([to_tensor(im)[0] for im in images]) if images else torch.zeros(0, 3, H, WMAX)
        if not len(xs):
            return []
        lg = self.m(xs); p = lg.softmax(-1).max(-1).values.mean(1).numpy()
        return list(zip(decode(lg), p.tolist()))
