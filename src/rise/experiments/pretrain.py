"""Contrastive pretraining of the *rāga*-independent *svara* encoder.

The encoder never sees a *svara* label. It sees a pitch contour and an augmented
view of the same contour, and learns to place the two close together and away from
every other contour in the batch. Because the augmentations imitate the way a
singer varies one *svara* between renditions, what survives that training is
whatever identifies a *svara* independently of how it happened to be sung.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from info_nce import InfoNCE
from torch.utils.data import DataLoader

from ..config import experiment_parameters
from ..console import console, detail, parameters_table, progress, rule
from ..data.datasets import ContourDataset, append_silence_mask
from ..data.preprocessing import dataset_dir
from ..data.splits import load_split
from ..dsp.augmentation import augment_batch
from ..nn.models import ContrastiveModel
from ..paths import ENCODER_CHECKPOINT, ensure_dir
from ..reproducibility import resolve_device

DESCRIPTION = "Pretrain the svara encoder on unannotated CMR pitch contours (InfoNCE)"

#: Factor by which the learning rate is cut when the loss plateaus.
LR_DECAY = 0.5

#: Floor below which the learning rate is not reduced further.
MIN_LEARNING_RATE = 1e-6

#: Improvement below which a step counts as a plateau.
PLATEAU_THRESHOLD = 1e-6

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=256, help="negatives per anchor, plus one")
    parser.add_argument("--depth", type=int, default=5, help="number of Inception blocks")
    parser.add_argument("--embed-dim", type=int, default=48, help="width of the embedding h")
    parser.add_argument("--out-dim", type=int, default=16, help="width of the projection z")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20, help="plateau epochs before the LR is cut")

def run(args: argparse.Namespace) -> None:
    rule("Contrastive pretraining")
    parameters_table(experiment_parameters(args))
    device = resolve_device(args.device)

    split = load_split(dataset_dir("pretrain"), "train")
    loader = DataLoader(ContourDataset(split["contours"]), batch_size=args.batch_size, shuffle=True)
    detail(f"{len(loader.dataset)} plausible svaras, {len(loader)} steps per epoch")

    model = ContrastiveModel(args.embed_dim, args.depth, args.out_dim).to(device)
    best_loss = train(model, loader, epochs=args.epochs, learning_rate=args.lr, patience=args.patience)

    console.print(f"Best InfoNCE loss: [metric]{best_loss:.6f}[/metric]")

def train(
    model: ContrastiveModel,
    loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> float:
    """Minimise the InfoNCE loss, checkpointing the encoder whenever it improves."""
    device = next(model.parameters()).device
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=LR_DECAY,
        patience=patience,
        threshold=PLATEAU_THRESHOLD,
        min_lr=MIN_LEARNING_RATE,
    )
    criterion = InfoNCE()
    best_loss = np.inf

    with progress() as bar:
        epoch_task = bar.add_task("Epochs", total=epochs)
        step_task = bar.add_task("Steps", total=len(loader), visible=False)

        for epoch in range(epochs):
            bar.update(step_task, completed=0, visible=True)
            model.train()
            running_loss = 0.0

            for batch in loader:
                anchors = batch.to(device)
                positives = augment_batch(anchors)
                loss = criterion(model(append_silence_mask(anchors)), model(append_silence_mask(positives)))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
                bar.advance(step_task)

            epoch_loss = running_loss / len(loader)
            scheduler.step(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                ensure_dir(ENCODER_CHECKPOINT.parent)
                torch.save(model.encoder.state_dict(), ENCODER_CHECKPOINT)

            bar.update(epoch_task, advance=1, description=f"Epoch {epoch + 1}/{epochs} · loss {epoch_loss:.6f}")
            bar.update(step_task, visible=False)

    return float(best_loss)
