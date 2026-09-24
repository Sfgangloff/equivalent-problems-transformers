"""Small shared helpers used across training and evaluation scripts."""

from __future__ import annotations

import torch


def pick_device(requested: str) -> torch.device:
    """Resolve a config's device string to a concrete `torch.device`.

    "auto" prefers CUDA, then Apple MPS, then falls back to CPU. Any other
    value (e.g. "cpu", "cuda:0") is passed through to `torch.device` as-is.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
