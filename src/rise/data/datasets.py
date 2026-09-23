"""Torch datasets and the input encoding they share.

Two shapes of observation cover every experiment: a single pitch contour, and a
*svara* together with the contours that precede and succeed it. Everything the
models consume is one of those two, so there are two datasets rather than one per
experiment.
"""

from __future__ import annotations

import numpy.typing as npt
import torch
from torch.utils.data import Dataset


def append_silence_mask(contours: torch.Tensor) -> torch.Tensor:
    """Turn ``(batch, 1, frames)`` of pitch into the ``(batch, 2, frames)`` model input.

    Unvoiced frames reach the model as NaN. They are replaced by zero — a value the
    convolutions can propagate — and a binary channel is appended marking where they
    were, so that the model can tell "silent" from "at the bottom of the pitch
    range" instead of having to guess.
    """
    mask = torch.isnan(contours).float()
    return torch.cat([torch.nan_to_num(contours, nan=0.0), mask], dim=1)


def as_contours(values: npt.ArrayLike) -> torch.Tensor:
    """Coerce padded contours to a float32 ``(n, frames)`` tensor."""
    if isinstance(values, torch.Tensor):
        return values.to(dtype=torch.float32)
    return torch.as_tensor(values, dtype=torch.float32)


class ContourDataset(Dataset):
    """A collection of padded pitch contours, one per item.

    Yields ``(1, frames)``: the channel axis is present but empty of the silence
    mask, which :func:`append_silence_mask` adds once the batch is on the device.
    """

    def __init__(self, contours: npt.ArrayLike) -> None:
        self.contours = as_contours(contours)

    def __len__(self) -> int:
        return len(self.contours)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.contours[index].unsqueeze(0)


class ContextualSvaraDataset(Dataset):
    """A *svara*, its preceding and succeeding context, and its label.

    Yields ``(prec, curr, succ, target)`` with each contour shaped ``(1, frames)``.
    The label is the *svara* for the classification experiment and the *svara*-form
    for the clustering experiment; the dataset itself does not distinguish them.
    """

    def __init__(
        self,
        preceding: npt.ArrayLike,
        current: npt.ArrayLike,
        succeeding: npt.ArrayLike,
        targets: npt.ArrayLike,
    ) -> None:
        self.preceding = as_contours(preceding)
        self.current = as_contours(current)
        self.succeeding = as_contours(succeeding)
        self.targets = torch.as_tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.preceding[index].unsqueeze(0),
            self.current[index].unsqueeze(0),
            self.succeeding[index].unsqueeze(0),
            self.targets[index],
        )

    @classmethod
    def from_split(cls, split: dict[str, torch.Tensor]) -> ContextualSvaraDataset:
        """Build the dataset from a split saved by the preprocessing experiment."""
        return cls(split["prec"], split["curr"], split["succ"], split["targets"])

    @property
    def num_classes(self) -> int:
        return int(torch.unique(self.targets).numel())
