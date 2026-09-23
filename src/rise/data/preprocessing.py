"""Turning raw pitch tracks into the tensors each experiment consumes.

Every dataset in the toolbox is built by the same three steps — clean the ``f0``
track, cut segments out of it, normalise and pad them — differing only in where
the segment boundaries come from. Those boundaries are the interesting part, and
each builder below documents its own:

* **classification / clustering** — expert *svara* annotations, plus a fixed window
  of melodic context on either side.
* **pretraining / synthesis** — beat and downbeat markers, from which plausible
  *svara* boundaries are sampled; no melodic annotation is used.
* **pattern recognition** — annotated melodic phrases, i.e. whole *svara*
  sequences rather than single *svaras*.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from ..console import artifact, banner, detail, progress, rule
from ..dsp.pitch import hz_to_cents, interpolate_gaps, normalise_cents, smooth_contour, unvoiced_to_nan
from ..paths import CACHE_DIR, ensure_dir
from .corpora import (
    CMR,
    IAMMS,
    RAGA_SVARAS,
    RAGAS,
    SVARAS,
    VARNAM,
    read_beats,
    read_elan_annotations,
    read_pitch_track,
    svara_of,
    varnam_performers,
    varnam_svara_forms,
)
from .splits import grouped_split, save_splits, stratified_split

#: Seconds of preceding and succeeding pitch retained as melodic context. Beyond
#: the two adjacent *svaras* the correlation with the current one falls away.
CONTEXT_DURATION = 0.5

#: Note durations sampled at each beat, as multiples of the inter-beat interval.
#: The metrical grid gives no note boundaries directly, so a plausible one is drawn
#: from the durations a *svara* commonly occupies within a *tāḷa* cycle.
BEAT_SUBDIVISIONS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

#: Shortest usable melodic phrase, in frames.
MIN_PHRASE_FRAMES = 10

#: A phrase flatter than this, in cents, carries no melodic movement to match on.
MIN_PHRASE_RANGE_CENTS = 50

#: A phrase whose residual about a straight line is this flat is a tracking
#: artefact — a glide or a constant — rather than a melodic contour.
MIN_PHRASE_RESIDUAL_CENTS = 10

#: IAMMS phrases whose pitch extraction was found to be faulty on inspection,
#: indexed in traversal order. Removing them takes the subset from 199 to 143.
FAULTY_IAMMS_PHRASES = frozenset({39, 40, 42, 48, 67, 68, 76, 83, 118, 143, 146, 148, 150})

FloatArray = npt.NDArray[np.float64]


def dataset_dir(name: str) -> Path:
    """Directory holding the train / validation / test splits of one dataset."""
    return CACHE_DIR / "datasets" / name




def prepare_contour(
    times: FloatArray,
    frequency: FloatArray,
    tonic: float,
    smoothing_factor: float,
    interpolation_gap: float,
) -> FloatArray:
    """Clean one raw ``f0`` track and express it in cents relative to ``tonic``.

    Smoothing is applied in Hz and the conversion to cents comes last, so that the
    spline sees the signal on the scale it was tracked on.
    """
    contour = unvoiced_to_nan(frequency)
    contour = interpolate_gaps(contour, np.nan, interpolation_gap)
    contour = smooth_contour(times, contour, smoothing_factor)
    return hz_to_cents(contour, tonic)


def slice_contour(
    times: FloatArray,
    contour: FloatArray,
    start: float,
    end: float,
    *,
    inclusive: bool = False,
) -> FloatArray:
    """Return the part of ``contour`` lying between ``start`` and ``end`` seconds."""
    if inclusive:
        return contour[(times >= start) & (times <= end)]
    return contour[(times > start) & (times < end)]


def pad_contours(contours: Iterable[FloatArray]) -> torch.Tensor:
    """Normalise contours to [0, 1] and right-pad them to a common length."""
    return torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(normalise_cents(contour), dtype=torch.float32) for contour in contours],
        batch_first=True,
    )


def sample_beat_segments(
    times: FloatArray,
    contour: FloatArray,
    beat_times: FloatArray,
) -> list[FloatArray]:
    """Cut plausible *svara* segments from a contour using its beat grid.

    Each beat opens a segment whose length is a randomly drawn multiple of that
    beat's own duration, so the segments follow the tempo of the performance rather
    than a fixed clock.
    """
    segments = []
    for index in range(len(beat_times) - 1):
        start = beat_times[index]
        end = start + (beat_times[index + 1] - start) * np.random.choice(BEAT_SUBDIVISIONS)
        if end > beat_times[-1]:
            break
        segments.append(slice_contour(times, contour, start, end, inclusive=True))
    return segments


@cache
def prepared_cmr_contour(
    path: Path,
    tonic: float,
    smoothing_factor: float,
    interpolation_gap: float,
) -> tuple[FloatArray, FloatArray]:
    """Frame times and cleaned contour of one CMR recording.

    Cleaning a CMR pitch track is the most expensive step in preprocessing, and the
    pretraining and synthesis datasets both draw from the same recordings. The
    result is cached so that the cleaning runs once per recording, while the segment
    sampling — which is random, and deliberately independent between the two
    datasets — still runs afresh for each.
    """
    times, frequency = read_pitch_track(path)
    return times, prepare_contour(times, frequency, tonic, smoothing_factor, interpolation_gap)


def cmr_segments(smoothing_factor: float, interpolation_gap: float, description: str) -> list[FloatArray]:
    """Sample plausible *svara* segments from every beat-annotated CMR recording."""
    tonics = CMR.tonics()
    pitch_track_paths = sorted(CMR.pitch_track_dir.glob("*.tsv"))
    segments: list[FloatArray] = []

    with progress() as bar:
        task = bar.add_task(description, total=len(pitch_track_paths))
        for path in pitch_track_paths:
            beat_path = CMR.beat_dir / path.name
            tonic = tonics.get(int(path.stem))
            if beat_path.exists() and tonic is not None and not np.isnan(tonic):
                times, contour = prepared_cmr_contour(
                    path, float(tonic), smoothing_factor, interpolation_gap
                )
                beat_times, _ = read_beats(beat_path)
                segments.extend(sample_beat_segments(times, contour, beat_times))
            bar.advance(task)

    return segments


def varnam_svara_segments(
    raga: str,
    performer: str,
    smoothing_factor: float,
    interpolation_gap: float,
) -> tuple[FloatArray, FloatArray, list[tuple[float, float, str]]]:
    """Return the prepared contour, its frame times, and the annotated *svaras*."""
    tonic = float(VARNAM.tonics()[performer])
    times, frequency = read_pitch_track(VARNAM.pitch_track_dir / raga / f"{performer}.tsv")
    contour = prepare_contour(times, frequency, tonic, smoothing_factor, interpolation_gap)
    annotations = read_elan_annotations(VARNAM.annotation_dir / raga / f"{performer}.tsv")
    events = list(zip(annotations["start"], annotations["end"], annotations["label"], strict=True))
    return times, contour, events


def with_context(
    times: FloatArray,
    contour: FloatArray,
    start: float,
    end: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cut a *svara* and the :data:`CONTEXT_DURATION` seconds either side of it."""
    return (
        slice_contour(times, contour, start - CONTEXT_DURATION, start),
        slice_contour(times, contour, start, end),
        slice_contour(times, contour, end, end + CONTEXT_DURATION),
    )




def build_classification_datasets(smoothing_factor: float, interpolation_gap: float) -> None:
    """One dataset per *rāga*: each annotated *svara* in its melodic context.

    The label is the index of the *svara* within its *rāga*'s own set, so a
    pentatonic *rāga* yields five classes and a heptatonic one seven.
    """
    rule("Svara classification")
    for raga in RAGAS:
        banner(raga.upper())
        labels = RAGA_SVARAS[raga]
        preceding, current, succeeding, targets = [], [], [], []

        performers = varnam_performers(raga)
        with progress() as bar:
            task = bar.add_task(f"Extracting {raga}", total=len(performers))
            for performer in performers:
                times, contour, events = varnam_svara_segments(
                    raga, performer, smoothing_factor, interpolation_gap
                )
                for start, end, annotation in events:
                    prec, curr, succ = with_context(times, contour, start, end)
                    preceding.append(prec)
                    current.append(curr)
                    succeeding.append(succ)
                    targets.append(labels.index(svara_of(annotation)))
                bar.advance(task)

        detail(f"{len(targets)} svaras across {len(set(targets))} of the {len(labels)} classes")
        data = {
            "prec": pad_contours(preceding),
            "curr": pad_contours(current),
            "succ": pad_contours(succeeding),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
        save_splits(data, dataset_dir(f"classification_{raga}"), stratified_split(len(targets), targets))


def build_clustering_dataset(smoothing_factor: float, interpolation_gap: float) -> None:
    """Every expert *svara*-form annotation, pooled across all seven *rāgas*.

    A *svara*-form is a (*svara*, cluster) pair — one characteristic *gamaka*
    realisation of one *svara* — and the split is by form, so that the forms in the
    test set are ones the model has never been trained on.
    """
    rule("Svara-form clustering")
    forms = varnam_svara_forms()
    preceding, current, succeeding, svaras, clusters = [], [], [], [], []

    with progress() as bar:
        task = bar.add_task("Extracting svara-forms", total=len(RAGAS))
        for raga in RAGAS:
            for performer in varnam_performers(raga):
                annotated = forms[(forms["raga"] == raga) & (forms["performer"] == performer)]
                if annotated.empty:
                    continue
                tonic = float(VARNAM.tonics()[performer])
                times, frequency = read_pitch_track(VARNAM.pitch_track_dir / raga / f"{performer}.tsv")
                contour = prepare_contour(times, frequency, tonic, smoothing_factor, interpolation_gap)

                for row in annotated.itertuples():
                    prec, curr, succ = with_context(times, contour, float(row.start), float(row.end))
                    preceding.append(prec)
                    current.append(curr)
                    succeeding.append(succ)
                    svaras.append(SVARAS.index(svara_of(row.svara)))
                    clusters.append(int(row.cluster))
            bar.advance(task)

    targets = encode_svara_forms(svaras, clusters)
    detail(f"{len(targets)} annotations spanning {len(set(targets.tolist()))} svara-forms")

    data = {
        "prec": pad_contours(preceding),
        "curr": pad_contours(current),
        "succ": pad_contours(succeeding),
        "targets": torch.tensor(targets, dtype=torch.long),
        "svaras": svaras,
        "clusters": clusters,
    }
    save_splits(data, dataset_dir("clustering"), grouped_split(targets))


def encode_svara_forms(svaras: Sequence[int], clusters: Sequence[int]) -> npt.NDArray[np.int64]:
    """Map each (*svara*, cluster) pair to a contiguous *svara*-form label."""
    pairs = list(zip(svaras, clusters, strict=True))
    vocabulary = sorted(set(pairs))
    return np.array([vocabulary.index(pair) for pair in pairs], dtype=np.int64)


def build_pattern_recognition_dataset(smoothing_factor: float, interpolation_gap: float) -> None:
    """Annotated melodic phrases from IAMMS, for phrase-level retrieval.

    Phrases too short, too flat, or too close to a straight line are dropped as
    pitch-tracking artefacts, and the phrases listed in
    :data:`FAULTY_IAMMS_PHRASES` are dropped after inspection.
    """
    rule("Melodic pattern recognition")
    tonics = IAMMS.tonics()
    sequences, phrase_ids = [], []
    phrase_index = 0

    annotation_paths = sorted(IAMMS.annotation_dir.glob("*.tsv"))
    with progress() as bar:
        task = bar.add_task("Extracting phrases", total=len(annotation_paths))
        for path in annotation_paths:
            tonic = tonics.get(path.stem)
            if tonic is None or np.isnan(tonic):
                bar.advance(task)
                continue

            times, frequency = read_pitch_track(IAMMS.pitch_track_dir / path.name)
            contour = prepare_contour(times, frequency, float(tonic), smoothing_factor, interpolation_gap)

            for start, end, label in read_elan_annotations(path).itertuples(index=False):
                phrase = slice_contour(times, contour, start, end, inclusive=True)
                phrase = phrase[~np.isnan(phrase)]
                if not is_usable_phrase(phrase):
                    continue
                if phrase_index not in FAULTY_IAMMS_PHRASES:
                    sequences.append(phrase)
                    phrase_ids.append(int(label))
                phrase_index += 1
            bar.advance(task)

    detail(f"{len(sequences)} phrases across {len(set(phrase_ids))} phrase identifiers")
    data = {"sequences": pad_contours(sequences), "ids": torch.tensor(phrase_ids, dtype=torch.long)}
    save_splits(data, dataset_dir("pattern_recognition"), stratified_split(len(phrase_ids), phrase_ids))


def is_usable_phrase(phrase: FloatArray) -> bool:
    """Reject phrases that carry no melodic movement a retrieval model could match."""
    if len(phrase) < MIN_PHRASE_FRAMES or np.ptp(phrase) < MIN_PHRASE_RANGE_CENTS:
        return False
    frames = np.arange(len(phrase))
    residual = phrase - np.polyval(np.polyfit(frames, phrase, 1), frames)
    return bool(np.std(residual) >= MIN_PHRASE_RESIDUAL_CENTS)


def build_pretrain_dataset(smoothing_factor: float, interpolation_gap: float) -> None:
    """Plausible *svaras* sampled from the beat grid of the unannotated CMR corpus."""
    rule("Contrastive pretraining")
    segments = cmr_segments(smoothing_factor, interpolation_gap, "Sampling plausible svaras")
    detail(f"{len(segments)} plausible svaras sampled from the CMR beat grid")
    save_splits({"contours": pad_contours(segments)}, dataset_dir("pretrain"), stratified_split(len(segments)))


def build_synthesis_dataset(smoothing_factor: float, interpolation_gap: float) -> None:
    """CMR segments to train the decoder on, and Varnam *svaras* to test it on.

    The two corpora are kept apart rather than split at random: the decoder is
    fitted on plausible segments and evaluated on genuinely annotated *svaras*, so
    that reconstruction quality is measured on the units the metrics describe.
    """
    rule("Svara synthesis")
    train_segments = cmr_segments(smoothing_factor, interpolation_gap, "Sampling CMR segments")

    test_segments: list[FloatArray] = []
    with progress() as bar:
        task = bar.add_task("Extracting Varnam svaras", total=len(RAGAS))
        for raga in RAGAS:
            for performer in varnam_performers(raga):
                times, contour, events = varnam_svara_segments(
                    raga, performer, smoothing_factor, interpolation_gap
                )
                test_segments.extend(slice_contour(times, contour, start, end) for start, end, _ in events)
            bar.advance(task)

    detail(f"{len(train_segments)} CMR segments for training, {len(test_segments)} Varnam svaras for testing")
    directory = ensure_dir(dataset_dir("synthesis"))
    torch.save({"train": pad_contours(train_segments), "test": pad_contours(test_segments)}, directory / "data.pt")
    artifact("Synthesis dataset", directory / "data.pt")
