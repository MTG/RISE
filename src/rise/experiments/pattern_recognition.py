"""Melodic pattern retrieval with the pretrained encoder.

The encoder was trained on single *svaras*; this experiment asks whether what it
learned survives at the phrase level. Each phrase is cut into windows of roughly
*svara* length, every window is embedded, and two phrases are compared by the mean
cosine similarity of their corresponding windows — so the comparison is made in the
space the encoder learned rather than on the raw contours.
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from ..config import experiment_parameters
from ..console import detail, metrics_table, parameters_table, progress, rule
from ..data.datasets import append_silence_mask
from ..data.preprocessing import dataset_dir
from ..data.splits import SPLIT_NAMES, load_split
from ..evaluation.retrieval import average_precision_per_query, retrieval_scores
from ..figures.plots import plot_grouped_distributions
from ..figures.style import save_figure
from ..nn.models import SvaraEmbedder
from ..paths import ENCODER_CHECKPOINT, RESULTS_DIR, ensure_dir
from ..reproducibility import resolve_device

DESCRIPTION = "Retrieve melodic patterns by cosine similarity of encoder embeddings"

RESULTS_FILE = "pattern_retrieval.tsv"

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="target window length in frames, about one svara at the pitch-track hop",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="windows embedded per forward pass")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", *SPLIT_NAMES],
        help="phrases to search over; nothing is trained here, so the default is every phrase",
    )

def run(args: argparse.Namespace) -> None:
    rule("Melodic pattern recognition")
    parameters_table(experiment_parameters(args))
    device = resolve_device(args.device)

    sequences, phrase_ids = load_phrases(args.split)
    detail(f"{len(sequences)} phrases across {len(set(phrase_ids.tolist()))} phrase identifiers")

    model = SvaraEmbedder(args.embed_dim, args.depth).to(device)
    model.encoder.load_state_dict(torch.load(ENCODER_CHECKPOINT, map_location=device))

    similarity = similarity_matrix(model, sequences, args.window_size, args.batch_size)
    scores = retrieval_scores(similarity, phrase_ids)

    report(scores)

    figure = plot_grouped_distributions(
        average_precision_per_query(similarity, phrase_ids),
        phrase_ids.tolist(),
        xlabel="Phrase ID",
        ylabel="MAP",
    )
    save_figure(figure, "pattern_average_precision")

def load_phrases(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the phrases to search over.

    Retrieval fits nothing, so holding phrases back would only shrink the pool a
    query is searched against; by default every phrase is used, which is also what
    the reported figures are computed over.
    """
    directory = dataset_dir("pattern_recognition")
    names = SPLIT_NAMES if split == "all" else (split,)
    parts = [load_split(directory, name) for name in names]
    return (
        torch.cat([part["sequences"] for part in parts]),
        torch.cat([part["ids"] for part in parts]),
    )

@torch.no_grad()
def similarity_matrix(
    model: SvaraEmbedder,
    sequences: torch.Tensor,
    window_size: int,
    batch_size: int,
) -> torch.Tensor:
    """Mean per-window cosine similarity between every pair of phrases.

    The sequences are padded to a common length, so every phrase splits into the
    same number of windows and each window can be embedded once rather than once per
    pair. Embeddings are L2-normalised, so the cosine similarity of two windows is
    their dot product and the whole matrix is one contraction.
    """
    model.eval()
    device = next(model.parameters()).device

    windows = split_into_windows(sequences, window_size)
    num_phrases, num_windows, length = windows.shape
    flattened = windows.reshape(-1, 1, length)

    embeddings = []
    with progress() as bar:
        task = bar.add_task("Embedding windows", total=len(flattened))
        for start in range(0, len(flattened), batch_size):
            batch = flattened[start : start + batch_size].to(device)
            embeddings.append(model(append_silence_mask(batch)).cpu())
            bar.advance(task, len(batch))

    embedded = torch.cat(embeddings).reshape(num_phrases, num_windows, -1)
    return torch.einsum("iwd,jwd->ij", embedded, embedded) / num_windows

def split_into_windows(sequences: torch.Tensor, window_size: int) -> torch.Tensor:
    """Cut every padded phrase into the same number of equal, non-overlapping windows."""
    length = sequences.shape[1]
    num_windows = max(length // window_size, 1)
    stride = length // num_windows
    return torch.stack([sequences[:, index * stride : (index + 1) * stride] for index in range(num_windows)], dim=1)

def report(scores) -> None:
    ensure_dir(RESULTS_DIR)
    row = scores.as_row()
    pd.DataFrame([row]).to_csv(RESULTS_DIR / RESULTS_FILE, sep="\t", index=False)
    metrics_table(
        "Melodic pattern recognition",
        ["Metric", "Score"],
        [
            ("Mean Average Precision (MAP)", f"{row['map']:.2%}"),
            ("Mean Reciprocal Rank (MRR)", f"{row['mrr']:.2%}"),
            *[(f"Precision@{k}", f"{value:.2%}") for k, value in scores.precision_at_k.items()],
        ],
    )
