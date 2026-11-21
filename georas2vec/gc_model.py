# Définitions du modèle GeoRas2Vec (extraites de train_final.py) pour l'inférence hors Kaggle.
import numpy as np, torch, torch.nn as nn
from scipy import ndimage as ndi
from numba import njit
import segmentation_models_pytorch as smp

CFG = dict(arch="unetpp", encoder="efficientnet-b4", input="rgb+lab",
           offsets=[(1, 0), (0, 1), (3, 0), (0, 3), (9, 0), (0, 9), (9, 9), (9, -9), (27, 0), (0, 27)],
           n_short=2, mws_stride=1, mws_min_size=40, tile=512)
OFFSETS, N_SHORT, K = CFG["offsets"], CFG["n_short"], len(CFG["offsets"])

# ------------------------------------------------------------------ affinités + mutex watershed
def fill_holes(inst):
    """Chaque zone sans polygone (mer, lac, halo) devient sa propre région."""
    holes, n = ndi.label(inst == 0)
    if n == 0:
        return inst
    out = inst.copy()
    out[holes > 0] = holes[holes > 0] + inst.max()
    return out


def affinity_targets(inst):
    H, W = inst.shape
    aff, mask = np.zeros((K, H, W), np.float32), np.zeros((K, H, W), np.float32)
    for k, (dy, dx) in enumerate(OFFSETS):
        ys, ye, xs, xe = max(0, -dy), H - max(0, dy), max(0, -dx), W - max(0, dx)
        a, b = inst[ys:ye, xs:xe], inst[ys + dy:ye + dy, xs + dx:xe + dx]
        aff[k, ys:ye, xs:xe] = a == b
        mask[k, ys:ye, xs:xe] = 1          # après fill_holes, tous les pixels ont une région
    return aff, mask


@njit
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit
def _mws_core(n, u, v, rep):
    parent = np.arange(n); size = np.ones(n, np.int64); m = len(u)
    mutex_a = np.empty(m, np.int64); mutex_b = np.empty(m, np.int64); nm = 0
    head = -np.ones(n, np.int64); nxt = np.empty(2 * m, np.int64); val = np.empty(2 * m, np.int64); nl = 0
    for e in range(m):
        ra, rb = _find(parent, u[e]), _find(parent, v[e])
        if ra == rb:
            continue
        if rep[e]:
            val[nl] = nm; nxt[nl] = head[ra]; head[ra] = nl; nl += 1
            val[nl] = nm; nxt[nl] = head[rb]; head[rb] = nl; nl += 1
            mutex_a[nm] = ra; mutex_b[nm] = rb; nm += 1
        else:
            small, other = (ra, rb) if size[ra] <= size[rb] else (rb, ra)
            blocked = False; p = head[small]
            while p != -1:
                k = val[p]; x = _find(parent, mutex_a[k]); y = _find(parent, mutex_b[k])
                if (x == small and y == other) or (x == other and y == small):
                    blocked = True; break
                p = nxt[p]
            if blocked:
                continue
            parent[small] = other; size[other] += size[small]; p = head[small]
            if p != -1:
                while nxt[p] != -1:
                    p = nxt[p]
                nxt[p] = head[other]; head[other] = head[small]
    out = np.empty(n, np.int64)
    for i in range(n):
        out[i] = _find(parent, i)
    return out


def mutex_watershed(aff, stride=CFG["mws_stride"], min_size=CFG["mws_min_size"]):
    _, H, W = aff.shape
    idx = np.arange(H * W).reshape(H, W)
    us, vs, ws, rep = [], [], [], []
    for k, (dy, dx) in enumerate(OFFSETS):
        a = idx[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        b = idx[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
        w = aff[k, max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)].copy()
        if k >= N_SHORT:
            a, b, w = a[::stride, ::stride], b[::stride, ::stride], 1 - w[::stride, ::stride]
        us.append(a.ravel()); vs.append(b.ravel()); ws.append(w.ravel()); rep.append(np.full(a.size, k >= N_SHORT))
    u, v, w, rep = map(np.concatenate, (us, vs, ws, rep))
    order = np.argsort(-w, kind="stable")
    lab = _mws_core(H * W, u[order], v[order], rep[order]).reshape(H, W)
    _, lab = np.unique(lab, return_inverse=True); lab = lab.reshape(H, W) + 1
    sizes = np.bincount(lab.ravel()); tiny = sizes < min_size; tiny[0] = False
    if tiny[lab].any():
        _, (iy, ix) = ndi.distance_transform_edt(tiny[lab], return_indices=True)
        lab = lab[iy, ix]; _, lab = np.unique(lab, return_inverse=True); lab = lab.reshape(H, W) + 1
    return lab.astype(np.int32)


# ------------------------------------------------------------------ modèle
def rgb2lab(x):
    """x : B x 3 x H x W dans [0,1] -> Lab normalisé (L/50-1, a/64, b/64)."""
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], device=x.device)
    xyz = torch.einsum("ij,bjhw->bihw", M, lin) / torch.tensor([0.95047, 1.0, 1.08883], device=x.device).view(1, 3, 1, 1)
    f = torch.where(xyz > 0.008856, xyz.clamp(min=1e-6) ** (1 / 3), 7.787 * xyz + 16 / 116)
    L, a, b = 116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L / 50 - 1, a / 64, b / 64], 1)


class GeoRas2VecNet(nn.Module):
    """Entrée : image uint8 B x 3 x H x W (H, W multiples de 32). Sortie : K affinités + 1 carte de failles (logits)."""

    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        Arch = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus}[cfg["arch"]]
        # v0.8: a frozen layer separator (colour plate + ink classes) feeds 10 extra channels to the network
        try:
            from .separator import Separator
        except ImportError:
            from separator import Separator
        self.sep = Separator() if cfg.get("sep") else None
        self.net = Arch(cfg["encoder"], encoder_weights="imagenet" if pretrained else None,
                        in_channels=16 if self.sep is not None else 6, classes=len(cfg["offsets"]) + 1)

    def forward(self, x8):
        with torch.autocast(x8.device.type, enabled=False):      # conversion de couleur en précision complète
            x = x8.float() / 255
            x = torch.cat([(x - self.mean) / self.std, rgb2lab(x)], 1)
            if self.sep is not None:
                with torch.no_grad():
                    plate, ink = self.sep.eval()(x8)
                x = torch.cat([x, (plate - 0.5) / 0.25, torch.softmax(ink.float(), 1)], 1)
        return self.net(x)


