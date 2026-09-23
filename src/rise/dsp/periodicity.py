"""Estimation of the oscillation count of a *gamaka*.

*Gamaka* is largely oscillatory, so the number of oscillations a contour contains
is a compact proxy for how much of that ornamentation a reconstruction preserved.
The estimate is the dominant non-DC bin of the magnitude spectrum, refined to
sub-bin resolution so that a contour of, say, 2.4 oscillations is not rounded to
the 2 or 3 the bin grid would otherwise force.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

#: Samples used to locate the maximum of the interpolant around the peak bin.
INTERPOLATION_RESOLUTION = 100


def oscillation_count(contour: npt.ArrayLike) -> float:
    """Return the estimated number of oscillations in ``contour``.

    NaN frames are dropped, the DC bin is zeroed so that the mean offset of the
    contour cannot win the arg-max, and the peak is refined by fitting a cubic
    spline through the peak bin and its two neighbours. A peak at either end of
    the spectrum has no such neighbourhood and is returned as the bin index.
    """
    samples = np.asarray(contour, dtype=np.float64).ravel()
    samples = samples[~np.isnan(samples)]
    if samples.size < 2:
        return 0.0

    spectrum = np.abs(np.fft.rfft(samples))
    spectrum[0] = 0.0
    peak = int(np.argmax(spectrum))
    if peak == 0 or peak == spectrum.size - 1:
        return float(peak)

    neighbourhood = CubicSpline([peak - 1, peak, peak + 1], spectrum[peak - 1 : peak + 2])
    grid = np.linspace(peak - 1, peak + 1, INTERPOLATION_RESOLUTION)
    return float(grid[np.argmax(neighbourhood(grid))])


def periodicity_error(reference: npt.ArrayLike, estimate: npt.ArrayLike) -> float:
    """Absolute difference in oscillation count between two contours."""
    return abs(oscillation_count(reference) - oscillation_count(estimate))
