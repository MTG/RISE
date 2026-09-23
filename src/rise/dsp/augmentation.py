"""Contrastive augmentations for pitch contours.

SimCLR needs a view of a *svara* that a human would still call the same *svara*.
The augmentations below are therefore modelled on the natural variability of
performance rather than on generic time-series noise: a singer takes a little
longer or a little less time over an ornament (temporal resizing), speeds up and
slows down within it (localised time warping), and drifts slightly in pitch
(magnitude warping).
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import torch

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*scipy\.ndimage\.filters.*")
    from tsaug import Drift, TimeWarp

#: Number of local speed changes applied by the time-warping augmentation.
NUM_SPEED_CHANGES = 5

#: Largest ratio between the fastest and the slowest warped segment.
MAX_SPEED_RATIO = 3.0

#: Largest pitch drift, as a fraction of the normalised pitch range.
MAX_DRIFT = 0.02

#: Fractional change in duration applied on top of the local warping.
DEFAULT_RESIZE_RANGE = 0.1

#: Sentinel written into padded regions of a batch; excluded from augmentation.
PAD_SENTINEL = -4200

#: Shortest run that :mod:`tsaug` can warp meaningfully.
MIN_WARPABLE_LENGTH = 4


def augment_batch(batch: torch.Tensor, resize_range: float = DEFAULT_RESIZE_RANGE) -> torch.Tensor:
    """Return a batch of augmented views, one per contour in ``batch``.

    ``batch`` is ``(N, 1, T)`` of normalised pitch, where 0 marks a padded or
    silent frame. Views are re-padded to a common length so that the result can be
    fed straight back into the encoder alongside the anchors.
    """
    contours = batch.clone()
    contours[contours == 0] = np.nan

    views = [perturb_contour(contour[0], resize_range) for contour in contours.cpu().numpy()]

    max_length = max(len(contour[0]) for contour in contours)
    padded = np.zeros((len(views), 1, max_length), dtype=np.float32)
    for index, view in enumerate(views):
        padded[index, 0, : len(view)] = view[:max_length]

    return torch.from_numpy(padded).to(batch.device)


def perturb_contour(contour: npt.NDArray, resize_range: float = DEFAULT_RESIZE_RANGE) -> npt.NDArray:
    """Apply the performance-like perturbations to one padded contour.

    Only the voiced span is perturbed; the padding on either side of it is carried
    through untouched so that the silence mask stays aligned with the pitch.
    """
    voiced = np.flatnonzero(~np.isnan(contour) & (contour != PAD_SENTINEL))
    start = int(voiced.min()) if voiced.size else 0
    stop = int(voiced.max()) if voiced.size else 0

    lead, span, trail = contour[:start], np.array(contour[start:stop]), contour[stop:]

    if len(span) > MIN_WARPABLE_LENGTH:
        span = TimeWarp(n_speed_change=NUM_SPEED_CHANGES, max_speed_ratio=MAX_SPEED_RATIO).augment(span)
        span = Drift(max_drift=MAX_DRIFT).augment(span)

    span = _resize(span, resize_range)
    return np.concatenate([lead, span, trail])


def _resize(span: npt.NDArray, resize_range: float) -> npt.NDArray:
    """Resample ``span`` to a duration within +/- ``resize_range`` of its own."""
    scale = np.random.uniform(1 - resize_range, 1 + resize_range)
    length = max(1, round(len(span) * scale))
    return np.interp(np.linspace(0, len(span) - 1, length), np.arange(len(span)), span)
