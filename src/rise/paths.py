"""Canonical filesystem layout of the project.

Every path used anywhere in the package is derived from :data:`PROJECT_ROOT`, so
that experiments can be run from any working directory and so that no module has
to hard-code a relative path of its own.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, i.e. the directory holding ``configs.yaml``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Read-only corpora shipped with the repository (pitch tracks, annotations, tonics).
DATA_DIR = PROJECT_ROOT / "data"

#: Preprocessed tensors, per-run weights and every other regenerable intermediate.
CACHE_DIR = PROJECT_ROOT / ".cache"

#: The two pretrained models, and nothing else. Weights that a downstream
#: experiment produces are intermediates and belong under :data:`CACHE_DIR`.
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

#: Metric tables written by the evaluation experiments.
RESULTS_DIR = PROJECT_ROOT / "results"

#: Publication-quality figures, as PNG at print resolution.
FIGURE_DIR = PROJECT_ROOT / "figures"

#: Default experiment configuration.
CONFIG_FILE = PROJECT_ROOT / "configs.yaml"

ENCODER_CHECKPOINT = CHECKPOINT_DIR / "encoder.pth"
DECODER_CHECKPOINT = CHECKPOINT_DIR / "decoder.pth"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and its parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
