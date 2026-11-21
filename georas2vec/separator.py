"""Layer separator of a printed map (v0.8).

A printed geological map is a stack of layers: the colour plate (units and their patterns) and several kinds of ink
(contacts, faults, coastlines, text, other lines, topographic base map). The separator predicts, for every pixel,
the colour plate as if no ink had been printed, and the class of ink printed on it. The segmentation network reads
the image together with these predictions, so it can tell a geological contact from a road or a contour line.

Ink classes: 0 none, 1 contact, 2 coastline, 3 fault, 4 text, 5 other line, 6 topographic base map.
"""
import torch, torch.nn as nn
import segmentation_models_pytorch as smp

NCLS = 7
CLASSES = ["none", "contact", "coastline", "fault", "text", "other_line", "topographic"]


class Separator(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=False):
        super().__init__()
        self.net = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=3 + NCLS)

    def forward(self, x8):
        """x8: uint8 B x 3 x H x W (H, W multiples of 32) -> (plate in [0, 1], ink-class logits)."""
        x = x8.float() / 255
        o = self.net((x - 0.5) / 0.25)
        return (x + torch.tanh(o[:, :3])).clamp(0, 1), o[:, 3:]
