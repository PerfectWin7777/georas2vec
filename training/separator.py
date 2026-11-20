"""Séparateur de couches d'une carte imprimée : image -> plaque couleur (sans encre) + classe d'encre par pixel.
Classes : 0 aucune, 1 contact, 2 côte, 3 faille, 4 texte, 5 autre trait, 6 fond topographique."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
NCLS = 7


class Separator(nn.Module):
    def __init__(self, encoder="resnet18", pretrained=True):
        super().__init__()
        self.net = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=3 + NCLS)

    def forward(self, x8):                       # x8 : uint8 B x 3 x H x W (H, W multiples de 32)
        x = x8.float() / 255
        o = self.net((x - 0.5) / 0.25)
        plate = (x + torch.tanh(o[:, :3])).clamp(0, 1)   # la plaque = image + correction (nulle hors encre)
        return plate, o[:, 3:]


def pad32(a):
    h, w = a.shape[:2]; H, W = -(-h // 32) * 32, -(-w // 32) * 32
    return np.pad(a, ((0, H - h), (0, W - w), (0, 0)), mode="edge"), (h, w)


@torch.no_grad()
def separate(model, rgb, tile=512):
    """rgb uint8 HxWx3 -> plaque uint8 HxWx3, probabilités d'encre NCLS x H x W."""
    a, (h, w) = pad32(rgb)
    x = torch.from_numpy(a.transpose(2, 0, 1).copy())[None]
    p, l = model.eval()(x)
    return (p[0].permute(1, 2, 0).numpy()[:h, :w] * 255).round().astype(np.uint8), torch.softmax(l[0], 0).numpy()[:, :h, :w]


def clean(rgb, plate, prob, remove=(4, 5, 6), thr=0.5):
    """Remplace par la plaque les pixels dont l'encre appartient aux classes à retirer (par défaut : texte, autres traits, topo)."""
    m = prob[list(remove)].sum(0) > thr
    out = rgb.copy(); out[m] = plate[m]
    return out
