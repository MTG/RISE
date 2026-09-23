"""Visual design for every figure in the thesis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from matplotlib import font_manager, rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure


rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Charter", "DejaVu Serif", "serif"]
rcParams["font.size"] = 8
rcParams["axes.titlesize"] = 8
rcParams["axes.labelsize"] = 7
rcParams["xtick.labelsize"] = 6
rcParams["ytick.labelsize"] = 6
rcParams["legend.fontsize"] = 6


INK = "#1a1a1a"
SECONDARY_INK = "#595959"
MUTED_INK = "#999999"
SURFACE = "#ffffff"
GRID = "#d0d0d0"

_ACE1 = "#0072b2"
_ACE2 = "#d55e00"
_ACE3 = "#009e73"
_ACE4 = "#cc79a7"
_ACE5 = "#e69f00"
_ACE6 = "#56b4e9"
_ACE7 = "#f0e442"
_ACE8 = "#8b0000"

categorical_palette = [_ACE1, _ACE2, _ACE3, _ACE4, _ACE5, _ACE6, _ACE7, _ACE8]


def categorical(n: int) -> list[str]:
    if n <= 0:
        return []
    return [categorical_palette[i % len(categorical_palette)] for i in range(n)]


BEAT = "#d55e00"
DOWNBEAT = "#1a1a1a"
HIGHLIGHT = "#0072b2"
HIGHLIGHT_ALPHA = 0.12

SEQUENTIAL = "Blues"


def figure_width(fraction: float = 1.0, column: str = "single") -> float:
    full = 6.1417 if column == "single" else 12.68
    return full * fraction


def despine(axes: Axes, *, left: bool = False, bottom: bool = False) -> None:
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    if left:
        axes.spines["left"].set_visible(False)
    if bottom:
        axes.spines["bottom"].set_visible(False)


def marker_cycle(n: int) -> list[str]:
    symbols = ["o", "s", "D", "^", "v", ">", "<", "p", "*", "h", "X"]
    return [symbols[i % len(symbols)] for i in range(n)]



def svarasthana_ticks(
    axes: Axes,
    svarasthanas: dict[str, int],
    *,
    octaves: list[int] | None = None,
    within: tuple[float, float] | None = None,
    min_gap_points: float = 30.0,
) -> None:
    if octaves is None:
        octaves = [0]

    labels, positions = [], []
    for octave in octaves:
        offset = octave * 1200
        for name, cents in svarasthanas.items():
            pos = cents + offset
            if within is not None and not (within[0] <= pos <= within[1]):
                continue
            labels.append(name)
            positions.append(pos)

    if not positions:
        return

    sorted_pairs = sorted(zip(positions, labels), key=lambda p: p[0])
    kept_positions, kept_labels = [sorted_pairs[0][0]], [sorted_pairs[0][1]]
    for pos, label in sorted_pairs[1:]:
        if (pos - kept_positions[-1]) >= min_gap_points:
            kept_positions.append(pos)
            kept_labels.append(label)

    axes.set_yticks(kept_positions, kept_labels)
    axes.tick_params(axis="y", length=2, pad=2, labelsize=6)



def apply_style() -> None:
    pass


def save_figure(figure: Figure, name: str, *, dpi: int = 300) -> None:
    from ..paths import FIGURE_DIR

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    import matplotlib.pyplot as plt
    plt.close(figure)
