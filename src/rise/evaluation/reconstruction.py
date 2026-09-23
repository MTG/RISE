"""Reconstruction metrics for *svara* synthesis.

Visual resemblance is not the question — three specific musical attributes are, and
each gets its own metric: the overall shape of the contour (DTW), the oscillation
of the *gamaka* (periodicity error), and the *svarasthāna*, that is, the pitch
position the *svara* occupies (pitch position error).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
import numpy.typing as npt

from ..dsp.periodicity import oscillation_count

FloatArray = npt.NDArray[np.float64]


@dataclass
class ReconstructionScores:
    """Per-*svara* errors, accumulated over the test set."""

    dtw_distances: list[float] = field(default_factory=list)
    periodicity_errors: list[float] = field(default_factory=list)
    pitch_position_errors: list[float] = field(default_factory=list)

    def add(self, reference: FloatArray, reconstruction: FloatArray) -> None:
        distance = dtw_distance(reference, reconstruction)
        if distance is not None:
            self.dtw_distances.append(distance)
        self.periodicity_errors.append(abs(oscillation_count(reference) - oscillation_count(reconstruction)))
        self.pitch_position_errors.append(pitch_position_error(reference, reconstruction))

    def means(self) -> dict[str, float]:
        return {
            "dtw_distance_cents": float(np.mean(self.dtw_distances)),
            "periodicity_error_oscillations": float(np.mean(self.periodicity_errors)),
            "pitch_position_error_cents": float(np.mean(self.pitch_position_errors)),
        }


def dtw_distance(reference: FloatArray, reconstruction: FloatArray) -> float | None:
    """Mean absolute deviation along the optimal DTW alignment, in cents.

    Dynamic time warping is used rather than a frame-wise distance so that a
    reconstruction that traces the right shape a little early or late is not
    penalised for the misalignment alone. Returns ``None`` when the two contours
    share no voiced frame.
    """
    voiced = ~np.isnan(reference) & ~np.isnan(reconstruction)
    if not voiced.any():
        return None
    left, right = reference[voiced], reconstruction[voiced]
    _, path = librosa.sequence.dtw(X=left, Y=right, metric="euclidean")
    return float(np.mean([abs(left[i] - right[j]) for i, j in path]))


def pitch_position_error(reference: FloatArray, reconstruction: FloatArray) -> float:
    """Absolute difference in median pitch, in cents.

    The median rather than the mean, because *gamaka* swings a contour well away
    from its *svarasthāna* and a mean would follow those excursions.
    """
    return float(abs(np.nanmedian(reference) - np.nanmedian(reconstruction)))
