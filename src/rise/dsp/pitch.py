"""Pitch-contour processing.

A *svara* is realised as a pitch contour rather than as a fixed pitch, so every
representation in this toolbox is built from an ``f0`` time series expressed in
cents relative to the tonic of the performer. This module holds the conversions
and the two cleaning steps — gap interpolation and spline smoothing — that turn a
raw FTA-Net pitch track into the model input described in the thesis.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import UnivariateSpline

#: Cents in one octave; the unit of the logarithmic pitch scale used throughout.
CENTS_PER_OCTAVE = 1200

#: Half-width, in cents, of the pitch range the models see: +/- 3.5 octaves around
#: the tonic, comfortably covering the range of a Carnatic vocalist.
CENTS_HALF_RANGE = 4200

#: Full width of that range, i.e. the divisor of the [0, 1] normalisation.
CENTS_FULL_RANGE = 2 * CENTS_HALF_RANGE

FloatArray = npt.NDArray[np.float64]


def hz_to_cents(frequency: npt.ArrayLike, tonic: float) -> FloatArray:
    """Convert an ``f0`` track in Hz to cents relative to ``tonic``.

    ``f_cent = 1200 * log2(f0 / f_tonic)``, so that the tonic (*sa*) sits at 0
    cents and a value is invariant to the absolute pitch the performer chose.
    """
    return CENTS_PER_OCTAVE * np.log2(np.asarray(frequency, dtype=float) / tonic)


def normalise_cents(cents: npt.ArrayLike) -> FloatArray:
    """Map the +/- :data:`CENTS_HALF_RANGE` band onto [0, 1] for the convolutions."""
    return (np.asarray(cents, dtype=float) + CENTS_HALF_RANGE) / CENTS_FULL_RANGE


def denormalise_cents(normalised: npt.ArrayLike) -> FloatArray:
    """Inverse of :func:`normalise_cents`, returning cents relative to the tonic."""
    return np.asarray(normalised, dtype=float) * CENTS_FULL_RANGE - CENTS_HALF_RANGE


def unvoiced_to_nan(frequency: npt.ArrayLike) -> FloatArray:
    """Replace the zeros the pitch tracker emits for unvoiced frames with NaN.

    NaN is the single representation of "no pitch here" used from this point on;
    it survives interpolation and smoothing and becomes the binary silence mask
    that is concatenated to the model input.
    """
    values = np.asarray(frequency, dtype=float).copy()
    values[values == 0] = np.nan
    return values


def interpolate_gaps(
    values: npt.ArrayLike,
    gap_value: float = np.nan,
    max_gap: float = 0.02,
    protected_indices: Iterable[int] | None = None,
) -> FloatArray:
    """Linearly interpolate across gaps in a pitch contour.

    A *gap* is a maximal run of frames equal to ``gap_value`` (or NaN when
    ``gap_value`` is NaN). Runs longer than ``max_gap``, and runs overlapping
    ``protected_indices``, are left for the caller to preserve; every other run is
    marked NaN and filled by linear interpolation, with the leading and trailing
    edges carried outwards.

    .. note::
       ``max_gap`` is expressed in frames while the shipped configuration supplies
       it in seconds (0.02), so with the default configuration *every* run is
       longer than the threshold and is therefore skipped. Because skipping merely
       leaves a NaN run as NaN, and the final pass interpolates all remaining NaNs,
       the net effect under the shipped configuration is that all gaps are filled.
       The published checkpoints were trained on contours produced this way, so the
       behaviour is preserved here verbatim.
    """
    contour = np.asarray(values, dtype=float).copy()
    protected = set(protected_indices or ())
    is_gap = np.isnan(contour) if np.isnan(gap_value) else contour == gap_value

    for start, stop in _runs(is_gap):
        if stop - start > max_gap:
            continue
        if protected & set(range(start, stop)):
            continue
        contour[start:stop] = np.nan

    return pd.Series(contour).interpolate(method="linear").ffill().bfill().to_numpy()


def smooth_contour(
    times: npt.ArrayLike,
    cents: npt.ArrayLike,
    smoothing_factor: float = 0.5,
    min_points: int = 4,
) -> FloatArray:
    """Smooth a pitch contour with a peak-preserving smoothing spline.

    Each voiced run is normalised to [0, 1] before fitting so that
    ``smoothing_factor`` — the residual budget of :class:`UnivariateSpline` — has
    the same meaning regardless of how wide that run's pitch excursion is, which
    is what keeps *gamaka* peaks from being flattened. Runs too short to fit a
    spline are passed through unchanged, and unvoiced frames stay NaN.
    """
    times = np.asarray(times, dtype=float)
    cents = np.asarray(cents, dtype=float)
    smoothed = np.full_like(cents, np.nan)

    voiced = np.where(~pd.isna(times) & ~pd.isna(cents))[0]
    if voiced.size == 0:
        return smoothed

    for run in np.split(voiced, np.where(np.diff(voiced) > 1)[0] + 1):
        run_times, run_cents = times[run], cents[run]
        if run.size >= min_points:
            low, high = run_cents.min(), run_cents.max()
            spline = UnivariateSpline(run_times, (run_cents - low) / (high - low), s=smoothing_factor)
            smoothed[run] = spline(run_times) * (high - low) + low
        elif run.size > 1:
            smoothed[run] = np.interp(run_times, run_times, run_cents)

    return smoothed


def _runs(flags: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return the ``[start, stop)`` bounds of every maximal run of ``True``."""
    padded = np.concatenate(([False], np.asarray(flags, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))
