"""The six experiments of the toolbox, plus the thesis figure generator.

Each module exposes ``DESCRIPTION``, ``add_arguments(parser)`` and ``run(args)``, so
that the command line is assembled from the modules themselves and adding an
experiment means adding one file.
"""

from . import (
    classification,
    clustering,
    pattern_recognition,
    preprocess,
    pretrain,
    synthesis,
    thesis_figures,
)

#: Experiments in the order they have to be run.
EXPERIMENTS = {
    "preprocess": preprocess,
    "pretrain": pretrain,
    "classification": classification,
    "clustering": clustering,
    "pattern_recognition": pattern_recognition,
    "synthesis": synthesis,
    "figures": thesis_figures,
}

__all__ = ["EXPERIMENTS"]
