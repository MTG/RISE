"""The three corpora, and the Carnatic vocabulary shared across them.

Each corpus contributes one thing the others cannot:

``CMR``   Carnatic Music Rhythm — no melodic annotation at all, but beat and
          downbeat markers from which plausible *svara* boundaries can be inferred.
          This is the unannotated corpus the encoder is pretrained on.
``Varnam``  Carnatic Varnam — *svara* and *svara*-form annotations across seven
          *rāgas*, used for supervised fine-tuning, classification and clustering.
``IAMMS``   Indian Art Music Melodic Similarity — labelled melodic phrases, used to
          test whether the representation transfers beyond the individual *svara*.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..paths import DATA_DIR

#: The seven *svaras* in *sargam* notation, in ascending order of the *ārōhaṇa*:
#: sa, ri, ga, ma, pa, dha, ni. Analogous to Western solfège.
SVARAS: tuple[str, ...] = ("S", "R", "G", "M", "P", "D", "N")

#: The *svaras* present in each *rāga* of the Carnatic Varnam dataset. Ābhōgi and
#: Mōhanam are *auḍava* (pentatonic) and omit two degrees; the rest are *sampūrṇa*
#: and carry all seven. The index of a *svara* in its rāga's tuple is its class
#: label, so these orderings are part of the data contract and must not change.
RAGA_SVARAS: dict[str, tuple[str, ...]] = {
    "abhogi": ("S", "R", "G", "M", "D"),
    "begada": SVARAS,
    "kalyani": SVARAS,
    "mohanam": ("S", "R", "G", "P", "D"),
    "sahana": SVARAS,
    "saveri": SVARAS,
    "sri": SVARAS,
}

#: The seven *rāgas*, in the order they are reported in the thesis.
RAGAS: tuple[str, ...] = tuple(RAGA_SVARAS)

#: Transliterated *rāga* names, for figures and tables.
RAGA_DISPLAY_NAMES: dict[str, str] = {
    "abhogi": "Ābhōgi",
    "begada": "Bēgaḍa",
    "kalyani": "Kalyāṇi",
    "mohanam": "Mōhanam",
    "sahana": "Sahāna",
    "saveri": "Sāvēri",
    "sri": "Śrī",
}

#: Pitch position (*svarasthāna*) of each *svara* of each *rāga*, in cents above the
#: tonic, with the variant of the degree the *rāga* takes named in the key. These are
#: the theoretical positions of the twelve-tone *dvādaśa svarasthāna* grid; a
#: performed *svara* oscillates around its position rather than sitting on it, which
#: is precisely what makes the contour, and not the position, the object of study.
RAGA_SVARASTHANA_CENTS: dict[str, dict[str, int]] = {
    "abhogi": {"S": 0, "R2": 200, "G2": 300, "M1": 500, "D2": 900},
    "begada": {"S": 0, "R2": 200, "G3": 400, "M1": 500, "P": 700, "D2": 900, "N3": 1100},
    "kalyani": {"S": 0, "R2": 200, "G3": 400, "M2": 600, "P": 700, "D2": 900, "N3": 1100},
    "mohanam": {"S": 0, "R2": 200, "G3": 400, "P": 700, "D2": 900},
    "sahana": {"S": 0, "R2": 200, "G3": 400, "M1": 500, "P": 700, "D2": 900, "N2": 1000},
    "saveri": {"S": 0, "R1": 100, "G3": 400, "M1": 500, "P": 700, "D1": 800, "N3": 1100},
    "sri": {"S": 0, "R2": 200, "G2": 300, "M1": 500, "P": 700, "D2": 900, "N2": 1000},
}

#: Full *sargam* syllables, for axis labels on pitch-track figures.
SVARA_DISPLAY_NAMES: dict[str, str] = {
    "S": "Sa",
    "R": "Ri",
    "G": "Ga",
    "M": "Ma",
    "P": "Pa",
    "D": "Dha",
    "N": "Ni",
}



@dataclass(frozen=True)
class Corpus:
    """Where one corpus keeps its pitch tracks, annotations and tonics."""

    name: str

    @property
    def root(self) -> Path:
        return DATA_DIR / self.name

    @property
    def pitch_track_dir(self) -> Path:
        return self.root / "pitch_tracks"

    @property
    def annotation_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def beat_dir(self) -> Path:
        return self.root / "beats"

    def tonics(self) -> dict[Any, float]:
        """Return the tonic frequency of every performance, keyed as in the corpus."""
        return _read_tonics(self.root / "tonics.yaml")


@cache
def _read_tonics(path: Path) -> dict[Any, float]:
    """Read and cache a tonics file; it is small, read often and never changes."""
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


CMR = Corpus("CMR")
VARNAM = Corpus("Varnam")
IAMMS = Corpus("IAMMS")




def read_pitch_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column ``time``/``frequency`` pitch track.

    Returns the frame times in seconds and the raw ``f0`` in Hz, with the zeros the
    tracker emits for unvoiced frames left in place for the caller to interpret.
    """
    track = pd.read_csv(path, sep="\t", header=None, names=["time", "frequency"])
    return track["time"].to_numpy(dtype=float), track["frequency"].to_numpy(dtype=float)


def read_beats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``time``/``beat`` annotation, returning beat times and positions.

    The beat position is the index of the beat within the *tāḷa* cycle, so a
    position of 1 marks a *sama* — the downbeat that opens a cycle.
    """
    beats = pd.read_csv(path, header=None, names=["time", "beat"])
    return beats["time"].to_numpy(dtype=float), beats["beat"].to_numpy(dtype=int)


def read_elan_annotations(path: Path) -> pd.DataFrame:
    """Read an ELAN export, returning ``start``, ``end`` (seconds) and ``label``.

    ELAN writes timestamps as ``HH:MM:SS.mmm``, sometimes without the hour field;
    both spellings are parsed to seconds here so that no caller has to.
    """
    annotations = pd.read_csv(path, sep="\t")
    return pd.DataFrame(
        {
            "start": annotations["Begin time"].map(parse_timestamp),
            "end": annotations["End time"].map(parse_timestamp),
            "label": annotations["Annotation"],
        }
    )


def parse_timestamp(timestamp: str) -> float:
    """Convert an ``[HH:]MM:SS.mmm`` timestamp to seconds."""
    fields = [float(field) for field in str(timestamp).split(":")]
    seconds = 0.0
    for field in fields:
        seconds = seconds * 60 + field
    return seconds


def svara_of(annotation: str) -> str:
    """Strip the octave marker from an annotation, leaving the *svara* symbol.

    An annotation may place the *svara* in an octave register (*sthāyi*) with a
    trailing ``^`` for the upper (*tāra*) or ``_`` for the lower (*mandra*). A
    *svara* keeps its identity across registers — ``"S^"`` and ``"S"`` are both
    *sa* — and the register is already carried by the pitch itself, so the marker
    is dropped.
    """
    return str(annotation)[0]


def varnam_performers(raga: str) -> list[str]:
    """Names of the performers who recorded the *varnam* of ``raga``, sorted."""
    return sorted(path.stem for path in (VARNAM.annotation_dir / raga).glob("*.tsv"))


def varnam_svara_forms() -> pd.DataFrame:
    """Expert *svara*-form annotations: which *gamaka* cluster each instance took."""
    return pd.read_csv(VARNAM.root / "svara_forms.tsv", sep=",")
