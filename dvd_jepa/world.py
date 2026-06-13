"""The world: a DVD logo bouncing in a box.

There are no actions and no agent. The logo just moves at constant speed and
reflects off the walls. This is deliberately the simplest non-trivial dynamical
system that still has the property that matters for a world model: the future
is *not* readable from a single frame (you need two frames to know the
velocity), but it *is* perfectly predictable once you do.

A single rendered "observation" handed to the model is a stack of two
consecutive frames, so velocity is observable.
"""
from __future__ import annotations
import numpy as np

H = 16      # frame height
W = 16      # frame width
EMB = 32    # embedding dimensionality (the model's entire world-state vector)
SIGMA = 1.5  # radius of the logo blob, in pixels


def render_blob(p, yy=None, xx=None):
    """Render a soft gaussian blob at position p=(y, x) into an HxW frame."""
    if yy is None:
        yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-(((yy - p[0]) ** 2 + (xx - p[1]) ** 2) / (2 * SIGMA ** 2))).astype(np.float32)


def make_sequences(n_seq=400, T=64, v=0.05, seed=None):
    """Generate a batch of bouncing-logo videos.

    Returns
    -------
    frames : float32 [n_seq, T, H, W] in [0, 1]
    pos    : float32 [n_seq, T, 2]  ground-truth (y, x) centre, for evaluation only
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pos = np.zeros((n_seq, T, 2), np.float32)
    frames = np.zeros((n_seq, T, H, W), np.float32)
    p = rng.uniform(2, H - 3, size=(n_seq, 2)).astype(np.float32)
    vel = (rng.choice([-1, 1], size=(n_seq, 2)) * v * (H - 1)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for t in range(T):
        p = p + vel
        for d, hi in ((0, H - 2), (1, W - 2)):
            flip = (p[:, d] > hi) | (p[:, d] < 1.0)
            vel[flip, d] *= -1
            p[:, d] = np.clip(p[:, d], 1.0, hi)
        pos[:, t] = p
        cy, cx = p[:, 0][:, None, None], p[:, 1][:, None, None]
        frames[:, t] = np.exp(-(((yy[None] - cy) ** 2 + (xx[None] - cx) ** 2) / (2 * SIGMA ** 2)))
    return frames, pos


def roll_one(T=44, teleport_at=None, seed=0, v=0.05):
    """A single video; optionally teleport the logo at one step to inject an
    anomaly the world model has no way to anticipate."""
    rng = np.random.RandomState(seed)
    p = rng.uniform(3, H - 4, size=2).astype(np.float32)
    vel = (rng.choice([-1, 1], size=2) * v * (H - 1)).astype(np.float32)
    frames = np.zeros((T, H, W), np.float32)
    pos = np.zeros((T, 2), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for t in range(T):
        if teleport_at is not None and t == teleport_at:
            p = rng.uniform(3, H - 4, size=2).astype(np.float32)
        p = p + vel
        for d, hi in ((0, H - 2), (1, W - 2)):
            if p[d] > hi or p[d] < 1.0:
                vel[d] *= -1
            p[d] = np.clip(p[d], 1.0, hi)
        pos[t] = p
        frames[t] = np.exp(-(((yy - p[0]) ** 2 + (xx - p[1]) ** 2) / (2 * SIGMA ** 2)))
    return frames, pos


def build_pairs(frames):
    """Turn videos into (observation, next-observation) latent-transition pairs.

    obs_t      = stack(frame_t,   frame_{t+1})   -> flattened 2*H*W vector
    next_obs_t = stack(frame_{t+1}, frame_{t+2})

    The model learns to predict the embedding of next_obs from the embedding of
    obs. The second half of each obs vector is exactly frame_{t+1}, which is
    what the decoder is asked to render.
    """
    import torch
    n, T = frames.shape[0], frames.shape[1]
    obs, nxt = [], []
    for t in range(T - 2):
        obs.append(frames[:, t:t + 2].reshape(n, -1))
        nxt.append(frames[:, t + 1:t + 3].reshape(n, -1))
    obs = torch.tensor(np.concatenate(obs), dtype=torch.float32)
    nxt = torch.tensor(np.concatenate(nxt), dtype=torch.float32)
    return obs, nxt
