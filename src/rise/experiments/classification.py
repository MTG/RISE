"""Svara classification, from scratch against fine-tuned from the pretrained encoder.

For each *rāga* of the Carnatic Varnam dataset the same contextual classifier is
trained twice — once with randomly initialised encoders, once with the pretrained
encoders adapted by LoRA — under identical hyperparameters and identical splits.
The gap between the two F1 scores is what the pretraining bought.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from ..config import experiment_parameters
from ..console import banner, detail, metrics_table, parameters_table, rule
from ..data.corpora import RAGA_DISPLAY_NAMES, RAGA_SVARAS, RAGAS, SVARA_DISPLAY_NAMES
from ..data.datasets import ContextualSvaraDataset
from ..data.preprocessing import dataset_dir
from ..data.splits import load_split
from ..figures.plots import plot_confusion_matrix
from ..figures.style import save_figure
from ..nn.lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK
from ..nn.models import ContextualSvaraClassifier
from ..paths import ENCODER_CHECKPOINT, RESULTS_DIR, ensure_dir
from ..reproducibility import resolve_device
from ..training import best_weights_path, load_checkpoint, predict, train_classifier

DESCRIPTION = "Fine-tune the pretrained encoder for svara classification, per rāga"

RESULTS_FILE = "classification_f1.tsv"

#: The two conditions compared, keyed by the tag used in file names and mapped to
#: the label used in tables and figures. Order is the order they are reported in.
CONDITIONS = {"scratch": "Fully supervised", "pretrained": "Semi-supervised"}

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--head-warmup-epochs",
        type=int,
        default=10,
        help="epochs during which only the classification head is trained",
    )
    parser.add_argument("--patience", type=int, default=30, help="early-stopping patience, in epochs")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--ragas", nargs="+", default=list(RAGAS), choices=list(RAGAS))

def run(args: argparse.Namespace) -> None:
    rule("Svara classification")
    parameters_table(experiment_parameters(args))
    device = resolve_device(args.device)
    scores: dict[str, dict[str, float]] = {}

    for raga in args.ragas:
        banner(RAGA_DISPLAY_NAMES[raga])
        loaders = build_loaders(raga, args.batch_size)
        labels = [SVARA_DISPLAY_NAMES[svara] for svara in RAGA_SVARAS[raga]]

        outcomes = {
            tag: evaluate_condition(raga, loaders, args, device, tag=tag) for tag in CONDITIONS
        }

        scores[raga] = {tag: outcome.f1 for tag, outcome in outcomes.items()}
        gain = outcomes["pretrained"].f1 - outcomes["scratch"].f1
        detail(f"F1 {outcomes['scratch'].f1:.4f} → {outcomes['pretrained'].f1:.4f} ({gain:+.4f})")

        for tag, outcome in outcomes.items():
            save_figure(plot_confusion_matrix(outcome.confusion, labels), f"confusion_{raga}_{tag}")

    report(scores)

@dataclass(frozen=True)
class Outcome:
    """The test-set F1 of one condition, and the confusion matrix behind it."""

    f1: float
    confusion: np.ndarray

def build_loaders(raga: str, batch_size: int) -> dict[str, DataLoader]:
    """Load the fixed splits of one *rāga* as data loaders."""
    directory = dataset_dir(f"classification_{raga}")
    return {
        name: DataLoader(
            ContextualSvaraDataset.from_split(load_split(directory, name)),
            batch_size=batch_size,
            shuffle=False,
            drop_last=name == "train",
        )
        for name in ("train", "val", "test")
    }

def evaluate_condition(
    raga: str,
    loaders: dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
    *,
    tag: str,
) -> Outcome:
    """Train one classifier for one *rāga* and score it on the held-out split."""
    pretrained = tag == "pretrained"
    num_classes = len(RAGA_SVARAS[raga])
    model = ContextualSvaraClassifier(args.embed_dim, args.depth, num_classes).to(device)

    if pretrained:
        model.load_pretrained_encoders(torch.load(ENCODER_CHECKPOINT, map_location=device))
        model.apply_lora(args.lora_rank, args.lora_alpha, args.lora_dropout)

    run_tag = f"classification_{raga}_{tag}"
    train_classifier(
        model,
        loaders["train"],
        loaders["val"],
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        head_warmup_epochs=args.head_warmup_epochs if pretrained else 0,
        run_tag=run_tag,
    )

    load_checkpoint(model, best_weights_path(run_tag))
    true, predicted = predict(model, loaders["test"])
    return Outcome(
        f1=float(f1_score(true, predicted, average="macro")),
        confusion=confusion_matrix(true, predicted, labels=range(num_classes)),
    )

def report(scores: dict[str, dict[str, float]]) -> None:
    """Write the score table and the figure that accompanies it."""
    if not scores:
        return

    frame = pd.DataFrame(
        [
            {
                "raga": raga,
                "f1_supervised": values["scratch"],
                "f1_semi_supervised": values["pretrained"],
                "difference": values["pretrained"] - values["scratch"],
            }
            for raga, values in scores.items()
        ]
    )
    ensure_dir(RESULTS_DIR)
    frame.to_csv(RESULTS_DIR / RESULTS_FILE, sep="\t", index=False)

    metrics_table(
        "Svara classification · macro F1",
        ["Rāga", *CONDITIONS.values(), "Δ"],
        [
            (
                RAGA_DISPLAY_NAMES[row.raga],
                f"{row.f1_supervised:.4f}",
                f"{row.f1_semi_supervised:.4f}",
                f"{row.difference:+.4f}",
            )
            for row in frame.itertuples()
        ],
    )
