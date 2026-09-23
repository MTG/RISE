"""Train / validation / test partitioning, and the on-disk form of a split.

Every experiment reads its data as three ``.pt`` files under one directory, so the
split is decided once during preprocessing and never re-drawn by a downstream
experiment. The seed is fixed for the same reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ..console import metrics_table
from ..paths import ensure_dir
from ..reproducibility import DEFAULT_SEED

#: Fraction of all observations held out for testing.
TEST_FRACTION = 0.4

#: Fraction of the training portion held out for validation.
VALIDATION_FRACTION = 0.3

#: In the *svara*-form experiment the split is by form rather than by observation,
#: and is even, so that half of the forms are never seen during training.
FORM_TEST_FRACTION = 0.5

SPLIT_NAMES = ("train", "val", "test")


def stratified_split(
    num_samples: int,
    stratify: npt.ArrayLike | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, npt.NDArray[np.int64]]:
    """Split indices into train / validation / test, preserving class proportions."""
    indices = np.arange(num_samples)
    train, test = train_test_split(indices, test_size=TEST_FRACTION, random_state=seed, stratify=stratify)
    train_stratify = np.asarray(stratify)[train] if stratify is not None else None
    train, validation = train_test_split(
        train, test_size=VALIDATION_FRACTION, random_state=seed, stratify=train_stratify
    )
    return {"train": train, "val": validation, "test": test}


def grouped_split(
    groups: npt.ArrayLike,
    seed: int = DEFAULT_SEED,
) -> dict[str, npt.NDArray[np.int64]]:
    """Split so that no group appears on both sides of the train / test boundary.

    Used for *svara*-form clustering, where the question is whether the encoder can
    group forms it has never seen; sharing a form between the splits would answer a
    different and much easier question.
    """
    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=FORM_TEST_FRACTION, random_state=seed)
    train, test = next(splitter.split(np.zeros(len(groups)), groups=groups))
    train, validation = train_test_split(
        train, test_size=VALIDATION_FRACTION, random_state=seed, stratify=groups[train]
    )
    return {"train": train, "val": validation, "test": test}


def save_splits(
    data: Mapping[str, Any],
    directory: Path,
    splits: Mapping[str, npt.NDArray[np.int64]],
    *,
    report: bool = True,
) -> None:
    """Write one ``.pt`` file per split, indexing every per-sample field of ``data``."""
    ensure_dir(directory)
    length = len(next(iter(data.values())))
    for name, indices in splits.items():
        torch.save({key: _take(value, indices, length) for key, value in data.items()}, directory / f"{name}.pt")
    if report:
        metrics_table(
            f"Split · {directory.name}",
            ["Split", "Samples"],
            [(name.capitalize(), str(len(indices))) for name, indices in splits.items()],
        )


def load_split(directory: Path, name: str) -> dict[str, Any]:
    """Read one split written by :func:`save_splits`."""
    return torch.load(directory / f"{name}.pt", weights_only=False)


def _take(value: Any, indices: npt.NDArray[np.int64], length: int) -> Any:
    """Index ``value`` by ``indices`` when it is per-sample, else pass it through."""
    if isinstance(value, torch.Tensor):
        return value[indices]
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == length:
        return [value[index] for index in indices]
    return value
