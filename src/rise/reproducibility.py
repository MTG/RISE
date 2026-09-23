"""Global determinism controls.

Called once from the command line entry point. Seeding every source of randomness
in one place — rather than at import time of a model module — keeps the seed an
explicit, overridable part of the experiment rather than a hidden side effect.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, and pin cuDNN to deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(preference: str | None = None) -> torch.device:
    """Return the device to run on, defaulting to CUDA when it is available."""
    if preference:
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
