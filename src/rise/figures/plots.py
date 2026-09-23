"""The figure vocabulary of the thesis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..dsp.pitch import CENTS_PER_OCTAVE
from .style import (
    GRID,
    HIGHLIGHT,
    HIGHLIGHT_ALPHA,
    INK,
    MUTED_INK,
    SECONDARY_INK,
    SEQUENTIAL,
    SURFACE,
    BEAT,
    DOWNBEAT,
    categorical,
    despine,
    figure_width,
    marker_cycle,
    svarasthana_ticks,
)

FloatArray = npt.NDArray[np.float64]

LABEL_HEADROOM = 0.22
PITCH_TRACK_ASPECT = 2.4


def _octaves_spanned(cents: FloatArray) -> list[int]:
    finite = np.asarray(cents)[np.isfinite(cents)]
    if finite.size == 0:
        return [0]
    lowest = int(np.floor(float(finite.min()) / CENTS_PER_OCTAVE))
    highest = int(np.floor(float(finite.max()) / CENTS_PER_OCTAVE))
    return list(range(lowest, highest + 1))


def _clean_axes(axes):
    despine(axes)
    axes.tick_params(axis="both", which="both", length=2, pad=2, labelsize=6)


def _grid_on(axes, *, axis="both"):
    axes.grid(True, axis=axis, color=GRID, linewidth=0.5, linestyle="-", alpha=0.7)
    axes.set_axisbelow(True)




def plot_pitch_track(
    times: FloatArray,
    cents: FloatArray,
    segments: Sequence[tuple[float, float, str]],
    svarasthanas: Mapping[str, int],
    *,
    highlight: str | None = None,
    width: float = figure_width(0.8),
) -> Figure:
    figure, axes = plt.subplots(figsize=(width, width / PITCH_TRACK_ASPECT), facecolor="white")

    span = (times >= segments[0][0]) & (times <= segments[-1][1])
    visible = cents[span]

    _grid_on(axes, axis="y")
    axes.plot(times[span], visible, color=INK, linewidth=1.2, solid_joinstyle="round")

    low, high = float(np.nanmin(visible)), float(np.nanmax(visible))
    margin = 0.06 * (high - low)
    axes.set_ylim(low - margin, high + margin + LABEL_HEADROOM * (high - low))

    for start, end, label in segments:
        is_hl = highlight is not None and label.upper() == highlight.upper()
        if is_hl:
            axes.axvspan(start, end, facecolor=HIGHLIGHT, alpha=HIGHLIGHT_ALPHA, linewidth=0, zorder=0)
        axes.axvline(start, color=MUTED_INK, linewidth=0.4, linestyle=(0, (3, 3)), zorder=1)
        axes.annotate(
            label,
            xy=((start + end) / 2, 0.94),
            xycoords=("data", "axes fraction"),
            ha="center", va="center",
            fontsize=7, fontweight="bold" if is_hl else "normal",
            color=INK if is_hl else SECONDARY_INK,
        )

    axes.axvline(segments[-1][1], color=MUTED_INK, linewidth=0.4, linestyle=(0, (3, 3)), zorder=1)
    axes.set_xlim(segments[0][0], segments[-1][1])

    svarasthana_ticks(
        axes, dict(svarasthanas),
        octaves=_octaves_spanned(visible),
        within=(low - margin, high + margin),
        min_gap_points=12,
    )
    _clean_axes(axes)
    figure.tight_layout()
    return figure




def plot_svara_renditions(
    renditions: Sequence[tuple[FloatArray, FloatArray]],
    svarasthanas: Mapping[str, int],
    *,
    columns: int = 2,
    width: float = figure_width(1.0),
) -> Figure:
    rows = int(np.ceil(len(renditions) / columns))
    figure, grid = plt.subplots(
        rows, columns, figsize=(width, 1.2 * rows), sharey=True, squeeze=False,
        facecolor="white",
    )

    all_cents = np.concatenate([cents for _, cents in renditions]) if renditions else np.array([0.0])
    lowest, highest = float(np.nanmin(all_cents)), float(np.nanmax(all_cents))
    span = highest - lowest
    axes_ymin = lowest - 0.08 * span
    axes_ymax = highest + 0.08 * span

    accent = categorical(len(renditions))

    for idx, (axes, (times, cents)) in enumerate(zip(grid.ravel(), renditions, strict=False)):
        t = times - times[0]
        axes.plot(t, cents, color=accent[idx], linewidth=1.1, alpha=0.8, solid_joinstyle="round")
        axes.set_ylim(axes_ymin, axes_ymax)
        axes.set_facecolor("white")
        _clean_axes(axes)
        _grid_on(axes)

    for axes in grid.ravel()[len(renditions) :]:
        axes.set_visible(False)

    figure.tight_layout(h_pad=0.5, w_pad=0.5)

    for axes in grid[:, 0]:
        svarasthana_ticks(
            axes, dict(svarasthanas),
            octaves=_octaves_spanned(all_cents),
            min_gap_points=12,
        )
    return figure




def plot_beat_grid(
    waveform: FloatArray,
    sample_rate: int,
    beat_times: FloatArray,
    beat_positions: npt.NDArray[np.int64],
    *,
    window: tuple[float, float] = (30.0, 40.0),
    width: float = figure_width(0.8),
) -> Figure:
    figure, axes = plt.subplots(figsize=(width, width / 3.1), facecolor="white")

    start, end = window
    samples = np.arange(int(start * sample_rate), min(int(end * sample_rate), len(waveform)))
    axes.plot(samples / sample_rate, waveform[samples], color=MUTED_INK, linewidth=0.3, alpha=0.9)

    inside = (beat_times >= start) & (beat_times <= end)
    for time, position in zip(beat_times[inside], beat_positions[inside], strict=True):
        is_downbeat = position == 1
        axes.axvline(
            time,
            color=DOWNBEAT if is_downbeat else BEAT,
            linewidth=0.8,
            linestyle=(0, (4, 2)) if is_downbeat else (0, (1, 2)),
            zorder=3,
        )

    axes.set_xlim(start, end)
    axes.set_yticks([])
    axes.grid(False)
    _clean_axes(axes)

    axes.legend(
        handles=[
            Line2D([], [], color=DOWNBEAT, linewidth=0.9, linestyle=(0, (4, 2)), label="Downbeats"),
            Line2D([], [], color=BEAT, linewidth=0.9, linestyle=(0, (1, 2)), label="Beats"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2, frameon=False, fontsize=6,
        handlelength=2.4, columnspacing=1.6, borderpad=0.0,
    )
    figure.tight_layout()
    return figure




def plot_grouped_distributions(
    values: Sequence[float],
    groups: Sequence,
    *,
    xlabel: str,
    ylabel: str,
    width: float = figure_width(0.9),
) -> Figure:
    order = sorted(set(groups))
    grouped = [np.asarray([v for v, g in zip(values, groups, strict=True) if g == key]) for key in order]

    figure, axes = plt.subplots(figsize=(width, width / 1.72), facecolor="white")

    _grid_on(axes, axis="y")
    box = axes.boxplot(
        grouped, widths=0.42, patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.2},
        whiskerprops={"color": MUTED_INK, "linewidth": 0.7},
        capprops={"color": MUTED_INK, "linewidth": 0.7},
        flierprops={
            "marker": "o", "markersize": 2,
            "markerfacecolor": "none", "markeredgecolor": MUTED_INK,
            "markeredgewidth": 0.4,
        },
    )
    for patch, colour in zip(box["boxes"], categorical(len(order)), strict=True):
        patch.set(facecolor=colour, alpha=0.8, edgecolor=INK, linewidth=0.8)

    axes.set_xticks(np.arange(1, len(order) + 1))
    _clean_axes(axes)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(
    matrix: npt.NDArray[np.int64],
    labels: Sequence[str],
    *,
    width: float = figure_width(0.55),
) -> Figure:
    matrix = np.asarray(matrix)
    totals = matrix.sum(axis=1, keepdims=True)
    proportion = np.divide(matrix, totals, out=np.zeros(matrix.shape, dtype=float), where=totals > 0)

    figure, axes = plt.subplots(figsize=(width, width), facecolor="white")
    axes.imshow(proportion, cmap=SEQUENTIAL, vmin=0, vmax=1)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes.text(
                column, row, str(matrix[row, column]),
                ha="center", va="center", fontsize=6,
                color=SURFACE if proportion[row, column] > 0.55 else INK,
            )

    axes.set_xticks(np.arange(len(labels)))
    axes.set_yticks(np.arange(len(labels)))
    axes.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    axes.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    axes.grid(which="major", visible=False)
    axes.grid(which="minor", color=SURFACE, linewidth=1)
    axes.tick_params(which="minor", length=0)
    axes.tick_params(axis="both", which="major", length=0, labelsize=6)
    despine(axes, left=True, bottom=True)
    figure.tight_layout()
    return figure


def plot_embedding_projections(
    panels: Sequence[Sequence[FloatArray]],
    labels: Sequence[Sequence[npt.NDArray[np.int64]]],
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    width: float = figure_width(0.8),
) -> Figure:
    rows, columns = len(row_labels), len(column_labels)
    figure, grid = plt.subplots(
        rows, columns, figsize=(width, width * rows / columns),
        squeeze=False, facecolor="white",
    )

    present = sorted({int(value) for row in labels for group in row for value in group if int(value) >= 0})
    colours = dict(zip(present, categorical(len(present)), strict=True))
    markers = dict(zip(present, marker_cycle(len(present)), strict=True))

    for row in range(rows):
        for column in range(columns):
            axes = grid[row][column]
            projection = np.asarray(panels[row][column])
            group_labels = np.asarray(labels[row][column], dtype=int)

            for label in sorted(set(group_labels.tolist())):
                selected = group_labels == label
                unclustered = label < 0
                axes.scatter(
                    projection[selected, 0], projection[selected, 1],
                    s=6,
                    c="none" if unclustered else colours[label],
                    marker="o" if unclustered else markers[label],
                    edgecolors=MUTED_INK if unclustered else SURFACE,
                    linewidths=0.3,
                    alpha=0.4 if unclustered else 0.9,
                )

            axes.set_xticks([])
            axes.set_yticks([])
            axes.grid(False)
            for spine in axes.spines.values():
                spine.set_visible(True)
                spine.set_color(GRID)

    figure.tight_layout()
    return figure


def plot_single_projection(
    projection: FloatArray,
    labels: npt.NDArray[np.int64],
    *,
    width: float = figure_width(0.45),
) -> Figure:
    """A single UMAP projection plot."""
    figure, axes = plt.subplots(figsize=(width, width), facecolor="white")

    unique = sorted(set(labels.tolist()))
    present = [u for u in unique if u >= 0]
    colours = dict(zip(present, categorical(len(present)), strict=True))

    for label in unique:
        selected = labels == label
        unclustered = label < 0
        axes.scatter(
            projection[selected, 0], projection[selected, 1],
            s=8,
            c="none" if unclustered else colours[label],
            marker="o",
            edgecolors=MUTED_INK if unclustered else colours.get(label, MUTED_INK),
            linewidths=0.4,
            alpha=0.5 if unclustered else 0.85,
        )

    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)

    figure.tight_layout()
    return figure
