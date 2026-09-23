"""Figure generation: the design system, the plot vocabulary, and the thesis figures."""

from .plots import (
    plot_beat_grid,
    plot_confusion_matrix,
    plot_embedding_projections,
    plot_grouped_distributions,
    plot_pitch_track,
    plot_svara_renditions,
)
from .style import apply_style, save_figure

__all__ = [
    "apply_style",
    "plot_beat_grid",
    "plot_confusion_matrix",
    "plot_embedding_projections",
    "plot_grouped_distributions",
    "plot_pitch_track",
    "plot_svara_renditions",
    "save_figure",
]
