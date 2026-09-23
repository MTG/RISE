r"""Regenerate the illustrative figures of the thesis from the corpora.

The result figures — the confusion matrices, the UMAP projection and the phrase-wise
average precision — are produced by the experiments that compute them, because they
depend on trained models. The three figures here illustrate the data itself and can
be drawn at any time.

Each is drawn against the caption it is placed with in the thesis, including the
width its ``\includegraphics`` gives it, so that the figure needs no rescaling and
its type comes out at the intended size on the page.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import experiment_parameters
from ..console import detail, rule, warn
from ..data.corpora import (
    CMR,
    RAGA_SVARASTHANA_CENTS,
    SVARA_DISPLAY_NAMES,
    VARNAM,
    read_beats,
    read_elan_annotations,
    read_pitch_track,
    svara_of,
    varnam_performers,
)
from ..data.preprocessing import prepare_contour
from ..figures.plots import plot_beat_grid, plot_pitch_track, plot_svara_renditions
from ..figures.style import figure_width, save_figure
from ..paths import CACHE_DIR, ensure_dir

DESCRIPTION = "Draw the illustrative figures of the thesis from the corpora"

#: Shortest usable rendition, in frames, for the variants figure.
MIN_RENDITION_FRAMES = 20

#: Widest excursion, in cents, a single *svara* is taken to span. *Gamaka* moves a
#: *svara* by a few *svarasthānas*, not by a fifth; a contour wider than this is an
#: octave error from the pitch tracker rather than an ornament, and picking it would
#: illustrate the tracker instead of the music.
MAX_RENDITION_RANGE_CENTS = 700

#: Recording whose beat grid illustrates how plausible *svara* boundaries are drawn.
#: A *khaṇḍa* cycle of five beats at 173 BPM, so the ten-second window the figure shows
#: holds six complete cycles.
BEAT_GRID_RECORDING = "13026"

#: Audio the beat grid falls back to when the CMR audio is not in the cache. The CMR
#: audio is licensed rather than redistributable, so it is not kept in the repository;
#: drop it into ``.cache`` and the figure picks it up on its own.
BEAT_GRID_YOUTUBE = "yqKoAt5a-k0"

#: Beats to a *tāḷa* cycle in the fallback performance — *rūpaka*, three to a cycle,
#: the same cycle length the CMR annotations of :data:`BEAT_GRID_RECORDING` carry.
BEAT_GRID_CYCLE = 3

#: The excerpt the pitch-track figure is drawn from: *rāga* Kalyāṇi as performed by
#: Ramakrishna Murthy, from the *ma* at 33.47 s to the *pa* ending at 37.07 s. Pinned
#: rather than searched for, so that the figure reproduces the one in the thesis
#: exactly; the number of realisations the caption claims is checked, not assumed.
PITCH_TRACK_PERFORMER = "ramakrishnamurthy"
PITCH_TRACK_WINDOW = (33.47, 37.07)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("names", nargs="*", default=[], help="figures to draw; omit for all of them")
    parser.add_argument("--list", action="store_true", help="list the available figures and exit")
    parser.add_argument("--smoothing-factor", type=float, default=0.5)
    parser.add_argument("--interpolation-gap", type=float, default=0.02)

    excerpt = parser.add_argument_group("pitch track")
    excerpt.add_argument("--raga", default="kalyani", help="rāga of the pitch-track excerpt")
    excerpt.add_argument("--performer", default=PITCH_TRACK_PERFORMER, help="performer of the excerpt")
    excerpt.add_argument("--svara", default="D", help="svara whose realisations are singled out")
    excerpt.add_argument("--start", type=float, default=PITCH_TRACK_WINDOW[0], help="excerpt start, in seconds")
    excerpt.add_argument("--end", type=float, default=PITCH_TRACK_WINDOW[1], help="excerpt end, in seconds")
    excerpt.add_argument(
        "--num-realisations",
        type=int,
        default=3,
        help="realisations of --svara the excerpt is expected to contain",
    )

    variants = parser.add_argument_group("svara variants")
    variants.add_argument("--variant-svara", default="G", help="svara whose variants are drawn")
    variants.add_argument("--num-variants", type=int, default=4)

    beats = parser.add_argument_group("beat grid")
    beats.add_argument("--audio", default=None, help="CMR audio file, if you hold the dataset")
    beats.add_argument(
        "--youtube",
        default=BEAT_GRID_YOUTUBE,
        help="video whose audio the figure falls back to, fetched into the cache",
    )


def run(args: argparse.Namespace) -> None:
    generators = figure_generators()
    if args.list:
        for name, generator in generators.items():
            detail(f"{name:<18} {(generator.__doc__ or '').strip().splitlines()[0]}")
        return

    rule("Thesis figures")
    detail(f"parameters: {experiment_parameters(args)}")
    for name in args.names or generators:
        if name not in generators:
            warn(f"Unknown figure {name!r}; run with --list to see the available names")
            continue
        generators[name](args)


def figure_generators() -> dict[str, Callable[[argparse.Namespace], None]]:
    return {
        "pitch-track": pitch_track_figure,
        "svara-variants": svara_variants_figure,
        "beat-grid": beat_grid_figure,
    }


def pitch_track_figure(args: argparse.Namespace) -> None:
    """A pitch track excerpt showing several distinct realisations of one svara."""
    times, contour = prepared_contour(args.raga, args.performer, args)
    annotations = read_elan_annotations(VARNAM.annotation_dir / args.raga / f"{args.performer}.tsv")
    events = [
        (row.start, row.end, SVARA_DISPLAY_NAMES[svara_of(row.label)]) for row in annotations.itertuples()
    ]

    highlight = SVARA_DISPLAY_NAMES[args.svara]
    excerpt = [event for event in events if args.start <= (event[0] + event[1]) / 2 <= args.end]
    if not excerpt:
        warn(f"No annotated svaras between {args.start}s and {args.end}s in {args.raga}/{args.performer}")
        return

    realisations = sum(label == highlight for _, _, label in excerpt)
    if realisations != args.num_realisations:
        warn(
            f"The excerpt holds {realisations} realisations of {highlight}, "
            f"not the {args.num_realisations} the caption claims"
        )

    figure = plot_pitch_track(
        times,
        contour,
        excerpt,
        RAGA_SVARASTHANA_CENTS[args.raga],
        highlight=highlight,
        width=figure_width(0.8),
    )
    save_figure(figure, "pitch_track")


def svara_variants_figure(args: argparse.Namespace) -> None:
    """Predominant pitch curves for several variants of one svara."""
    renditions: list[tuple[np.ndarray, np.ndarray]] = []

    for performer in varnam_performers(args.raga):
        times, contour = prepared_contour(args.raga, performer, args)
        annotations = read_elan_annotations(VARNAM.annotation_dir / args.raga / f"{performer}.tsv")
        matching = annotations[annotations["label"].map(svara_of) == args.variant_svara]

        candidates = []
        for row in matching.itertuples():
            window = (times >= row.start) & (times <= row.end)
            if window.sum() < MIN_RENDITION_FRAMES or not np.isfinite(contour[window]).all():
                continue
            excursion = float(np.ptp(contour[window]))
            if excursion <= MAX_RENDITION_RANGE_CENTS:
                candidates.append((excursion, times[window], contour[window]))
        if candidates:
            _, window_times, window_cents = max(candidates, key=lambda candidate: candidate[0])
            renditions.append((window_times, window_cents))
        if len(renditions) == args.num_variants:
            break

    if len(renditions) < args.num_variants:
        warn(f"Only {len(renditions)} clean renditions of {args.variant_svara} found in {args.raga}")
    if not renditions:
        return

    figure = plot_svara_renditions(
        renditions, RAGA_SVARASTHANA_CENTS[args.raga], width=figure_width(1.0)
    )
    save_figure(figure, "svara_variants")


def beat_grid_figure(args: argparse.Namespace) -> None:
    """A waveform excerpt with the beat and downbeat grid that segments it."""
    import librosa

    audio = args.audio or cmr_audio(BEAT_GRID_RECORDING)
    if audio:
        detail(f"Beat grid from {audio.name if isinstance(audio, Path) else audio}, with the CMR annotations")
        waveform, sample_rate = librosa.load(audio, sr=None, mono=True)
        beat_times, beat_positions = read_beats(CMR.beat_dir / f"{BEAT_GRID_RECORDING}.tsv")
    else:
        waveform, sample_rate = librosa.load(cached_audio(args.youtube), sr=None, mono=True)
        beat_times, beat_positions = tracked_beats(waveform, sample_rate)

    figure = plot_beat_grid(waveform, sample_rate, beat_times, beat_positions, width=figure_width(0.8))
    save_figure(figure, "beat_grid")


def cmr_audio(recording: str) -> Path | None:
    """The CMR audio for ``recording`` if it has been placed in the cache.

    The dataset is licensed and cannot ship with the repository, but its annotations
    can: dropping the audio into ``.cache`` is all that is needed for the figure to be
    drawn from the master its beats were tapped against.
    """
    for directory in (CACHE_DIR / "audio", CACHE_DIR):
        matches = sorted(directory.glob(f"{recording}*.wav")) if directory.exists() else []
        if matches:
            return matches[0]
    return None


def cached_audio(video_id: str) -> Path:
    """Fetch the audio of ``video_id`` into the cache once, and return where it landed."""
    path = ensure_dir(CACHE_DIR / "audio") / f"{video_id}.wav"
    if path.exists():
        return path

    try:
        from yt_dlp import YoutubeDL
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "fetching the beat-grid audio needs yt-dlp: pip install 'rise[audio]'"
        ) from None

    detail(f"Fetching audio for {video_id} into {path.parent}")
    options = {
        "format": "bestaudio/best",
        "outtmpl": f"{path.with_suffix('')}.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(options) as downloader:
        downloader.download([f"https://www.youtube.com/watch?v={video_id}"])
    return path


def tracked_beats(waveform, sample_rate: int, cycle: int = BEAT_GRID_CYCLE):
    """Beats estimated from the audio itself, grouped into cycles of ``cycle``.

    A beat tracker recovers the pulse but not which pulse begins the cycle, so the
    first beat it finds is taken as a *sama*. The figure illustrates what a beat grid
    looks like over a waveform; the experiments read their grid from the CMR
    annotations and never from this.
    """
    import librosa

    _, frames = librosa.beat.beat_track(y=waveform, sr=sample_rate, units="frames")
    times = librosa.frames_to_time(frames, sr=sample_rate)
    return times, np.arange(len(times)) % cycle + 1


def prepared_contour(raga: str, performer: str, args: argparse.Namespace):
    times, frequency = read_pitch_track(VARNAM.pitch_track_dir / raga / f"{performer}.tsv")
    tonic = float(VARNAM.tonics()[performer])
    return times, prepare_contour(times, frequency, tonic, args.smoothing_factor, args.interpolation_gap)
