"""The three small networks that make up DVD-JEPA.

Everything is a plain MLP so the entire forward pass is a handful of matrix
multiplies -- which is what lets us re-implement inference in ~40 lines of
JavaScript for the browser demo (see web/).

    Encoder    obs (2*H*W)  -> z (EMB)
    Predictor  z (EMB)      -> z_hat (EMB)        the world model / transition fn
    Decoder    z (EMB)      -> frame (H*W)        the optional "readout" head

The Encoder is used twice: once as the trainable *online* encoder and once as
an Exponential-Moving-Average *target* encoder whose output is the prediction
target (stop-gradient). That asymmetry, plus the variance term below, is what
stops the representation from collapsing to a constant.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import H, W, EMB


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * H * W, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, EMB),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """The world model proper: advance one step in representation space."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB, 64), nn.GELU(),
            nn.Linear(64, EMB),
        )

    def forward(self, z):
        return self.net(z)


class Decoder(nn.Module):
    """Optional readout: turn a latent back into a pixel frame. A pure JEPA has
    no decoder; this is what makes the dream visible (and useful)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB, 64), nn.GELU(),
            nn.Linear(64, 256), nn.GELU(),
            nn.Linear(256, H * W), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


def variance_term(z, eps=1e-4):
    """VICReg-style hinge: encourage every embedding dimension to keep a
    standard deviation of at least 1 across the batch. This is the explicit
    anti-collapse pressure; with it the emb-std stays well above zero."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()
