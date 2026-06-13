"""DVD-JEPA: a minimal, fully-reproducible Joint-Embedding Predictive
Architecture world model, trained on a bouncing DVD logo.

The package is intentionally tiny and dependency-light so the whole idea fits
in your head and trains on a CPU in under a minute. See README.md for the
paper-style writeup.
"""
from .world import H, W, EMB, make_sequences, roll_one, build_pairs, render_blob
from .models import Encoder, Predictor, Decoder, variance_term

__all__ = [
    "H", "W", "EMB",
    "make_sequences", "roll_one", "build_pairs", "render_blob",
    "Encoder", "Predictor", "Decoder", "variance_term",
]
__version__ = "0.1.0"
