"""Signal processing for pitch contours: conversion, cleaning, augmentation, periodicity."""

from .augmentation import augment_batch, perturb_contour
from .periodicity import oscillation_count, periodicity_error
from .pitch import (
    CENTS_FULL_RANGE,
    CENTS_HALF_RANGE,
    CENTS_PER_OCTAVE,
    denormalise_cents,
    hz_to_cents,
    interpolate_gaps,
    normalise_cents,
    smooth_contour,
    unvoiced_to_nan,
)

__all__ = [
    "CENTS_FULL_RANGE",
    "CENTS_HALF_RANGE",
    "CENTS_PER_OCTAVE",
    "augment_batch",
    "denormalise_cents",
    "hz_to_cents",
    "interpolate_gaps",
    "normalise_cents",
    "oscillation_count",
    "periodicity_error",
    "perturb_contour",
    "smooth_contour",
    "unvoiced_to_nan",
]
