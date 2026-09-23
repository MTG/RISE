"""RISE — a rāga-independent encoder for svara representation in Carnatic music.

A *svara* in Carnatic music is not a fixed pitch but an ornamented pitch contour
whose shape depends on the *svaras* around it, and annotated data for it is scarce.
This package learns a representation of that contour in two complementary ways:
by pretraining an encoder contrastively on unannotated recordings, and by giving the
downstream models the melodic context a *svara* was performed in.

The package is laid out along the pipeline:

``rise.data``
    corpora, preprocessing, torch datasets and the fixed train/validation/test splits
``rise.dsp``
    pitch-contour conversion, cleaning, augmentation and periodicity estimation
``rise.nn``
    the InceptionTime backbone, co-attention, LoRA and the three task models
``rise.evaluation``
    the retrieval, reconstruction and clustering metrics
``rise.experiments``
    the six experiments, each of which is one sub-command of the ``rise`` CLI
``rise.figures``
    the figure design system and the plots it produces
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
