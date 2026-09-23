"""Svara-form clustering in the learned embedding space.

A *svara*-form is one characteristic *gamaka* realisation of one *svara*. The
question here is not whether the model can label forms it was trained on, but
whether forms it has never seen still land together in the embedding space — so the
split is by form, and the metric is the agreement between HDBSCAN's clusters and the
expert annotations rather than any classification accuracy.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from torch.utils.data import DataLoader

from ..config import experiment_parameters
from ..console import banner, detail, metrics_table, parameters_table, rule
from ..data.datasets import ContextualSvaraDataset
from ..data.preprocessing import dataset_dir
from ..data.splits import load_split
from ..evaluation import normalised_mutual_information
from ..figures.style import save_figure
from ..nn.lora import DEFAULT_ALPHA, DEFAULT_DROPOUT
from ..nn.models import ContextualSvaraClassifier
from ..paths import ENCODER_CHECKPOINT, RESULTS_DIR, ensure_dir
from ..reproducibility import DEFAULT_SEED, resolve_device
from ..training import best_weights_path, encode, load_checkpoint, train_classifier

DESCRIPTION = "Cluster svara-form embeddings and score them against expert annotations"

RESULTS_FILE = "clustering_nmi.tsv"

#: LoRA rank used here; lower than for classification because the *svara*-form
#: task has fewer annotations per class to estimate the update from.
LORA_RANK = 4

CONDITIONS = {"scratch": "Fully supervised", "pretrained": "Semi-supervised"}

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--head-warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_DROPOUT)

def run(args: argparse.Namespace) -> None:
    rule("Svara-form clustering")
    parameters_table(experiment_parameters(args))
    device = resolve_device(args.device)

    directory = dataset_dir("clustering")
    splits = {name: load_split(directory, name) for name in ("train", "val", "test")}
    num_classes = int(max(int(split["targets"].max()) for split in splits.values())) + 1
    loaders = {
        name: DataLoader(
            ContextualSvaraDataset.from_split(split), batch_size=args.batch_size, shuffle=False
        )
        for name, split in splits.items()
    }
    truth = splits["test"]["targets"].numpy()
    detail(
        f"{len(splits['train']['targets'])} training and {len(truth)} test annotations "
        f"across {num_classes} svara-forms, disjoint between the two"
    )

    results, panels = {}, []
    for tag, condition in CONDITIONS.items():
        banner(condition)
        embeddings = fit_and_embed(loaders, num_classes, args, device, tag=tag)
        clusters = HDBSCAN().fit_predict(embeddings)
        results[tag] = float(normalised_mutual_information(truth, clusters))
        detail(f"NMI {results[tag]:.4f} over {len(set(clusters)) - (-1 in clusters)} clusters")
        panels.append((condition, embeddings, clusters))

    report(results)
    draw_projection(panels, truth)

def fit_and_embed(
    loaders: dict[str, DataLoader],
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    tag: str,
) -> np.ndarray:
    """Train the contextual model, then discard its head and keep the embeddings."""
    pretrained = tag == "pretrained"
    model = ContextualSvaraClassifier(args.embed_dim, args.depth, num_classes).to(device)

    if pretrained:
        model.load_pretrained_encoders(torch.load(ENCODER_CHECKPOINT, map_location=device))
        model.apply_lora(args.lora_rank, args.lora_alpha, args.lora_dropout)

    run_tag = f"clustering_{tag}"
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
    return encode(model, loaders["test"])

def report(results: dict[str, float]) -> None:
    ensure_dir(RESULTS_DIR)
    difference = results["pretrained"] - results["scratch"]
    pd.DataFrame(
        [
            {
                "nmi_supervised": results["scratch"],
                "nmi_semi_supervised": results["pretrained"],
                "difference": difference,
            }
        ]
    ).to_csv(RESULTS_DIR / RESULTS_FILE, sep="\t", index=False)

    metrics_table(
        "Svara-form clustering · normalised mutual information",
        ["Condition", "NMI"],
        [(CONDITIONS[tag], f"{value:.4f}") for tag, value in results.items()] + [("Δ", f"{difference:+.4f}")],
    )

def draw_projection(panels: list[tuple[str, np.ndarray, np.ndarray]], truth: np.ndarray) -> None:
    """Project both embedding spaces to two dimensions, one plot per condition."""
    try:
        from umap import UMAP
    except ImportError:
        detail("umap-learn is not installed; skipping the projection figure")
        return

    from ..figures.plots import plot_single_projection

    for condition, embeddings, clusters in panels:
        projection = UMAP(random_state=DEFAULT_SEED).fit_transform(embeddings)
        save_figure(plot_single_projection(projection, clusters), f"clustering_umap_{condition}_clusters")
        save_figure(plot_single_projection(projection, truth), f"clustering_umap_{condition}_truth")
