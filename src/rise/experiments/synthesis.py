"""Svara synthesis: reconstructing a pitch contour from its embedding.

The encoder is frozen and a transposed-Inception decoder is trained to invert it.
What the reconstruction preserves is what the embedding kept, so the evaluation
asks after three specific musical attributes rather than after visual resemblance:
the shape of the contour (DTW), the oscillation of the *gamaka* (periodicity error)
and the *svarasthāna* it occupies (pitch position error).

The decoder is fitted on plausible segments from the unannotated CMR corpus and
evaluated on annotated Varnam *svaras*, so the metrics describe genuine *svaras*.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import experiment_parameters
from ..console import detail, metrics_table, parameters_table, progress, rule
from ..data.datasets import ContourDataset, append_silence_mask
from ..data.preprocessing import dataset_dir
from ..dsp.pitch import denormalise_cents
from ..evaluation.reconstruction import ReconstructionScores
from ..nn.inception import InceptionDecoder, InceptionEncoder
from ..paths import DECODER_CHECKPOINT, ENCODER_CHECKPOINT, RESULTS_DIR, ensure_dir
from ..reproducibility import resolve_device

DESCRIPTION = "Reconstruct svara pitch contours from the frozen encoder's embeddings"

RESULTS_FILE = "synthesis_reconstruction.tsv"

#: Threshold above which the decoder's voicing head is taken to mean "voiced".
VOICING_THRESHOLD = 0.5

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="fit the decoder even when a checkpoint already exists",
    )

def run(args: argparse.Namespace) -> None:
    rule("Svara synthesis")
    parameters_table(experiment_parameters(args))
    device = resolve_device(args.device)

    encoder = InceptionEncoder(args.embed_dim, args.depth).to(device)
    encoder.load_state_dict(torch.load(ENCODER_CHECKPOINT, map_location=device))
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    data = torch.load(dataset_dir("synthesis") / "data.pt", weights_only=False)
    detail(f"{len(data['train'])} CMR segments for fitting, {len(data['test'])} Varnam svaras for testing")

    decoder = InceptionDecoder(args.embed_dim, args.depth).to(device)
    if DECODER_CHECKPOINT.exists() and not args.retrain:
        decoder.load_state_dict(torch.load(DECODER_CHECKPOINT, map_location=device))
        detail(f"Using the decoder checkpoint at {DECODER_CHECKPOINT}")
    else:
        loader = build_loader(encoder, data["train"], args.embed_batch_size, args.batch_size, device)
        loss = train(decoder, loader, epochs=args.epochs, learning_rate=args.lr, patience=args.patience)
        detail(f"Best reconstruction MSE {loss:.8f}")

    loader = build_loader(encoder, data["test"], args.embed_batch_size, args.batch_size, device)
    reconstructions = reconstruct(decoder, loader, data["test"].shape[1])
    evaluate(data["test"], reconstructions)

@torch.no_grad()
def embed(encoder: InceptionEncoder, contours: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Encode every contour, keeping the time axis for the decoder to expand."""
    loader = DataLoader(ContourDataset(contours), batch_size=batch_size, shuffle=False)
    return torch.cat([encoder(append_silence_mask(batch.to(device))).cpu() for batch in loader])

def build_loader(
    encoder: InceptionEncoder,
    contours: torch.Tensor,
    embed_batch_size: int,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    """Pair each embedding with the contour and voicing mask it has to reproduce."""
    embeddings = embed(encoder, contours, embed_batch_size, device)
    targets = contours.unsqueeze(1)
    voicing = (~torch.isnan(targets)).float()
    targets = torch.cat([torch.nan_to_num(targets, nan=0.0), voicing], dim=1)
    return DataLoader(TensorDataset(embeddings, targets), batch_size=batch_size)

def train(
    decoder: InceptionDecoder,
    loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> float:
    """Fit the decoder by mean squared error on pitch and voicing jointly."""
    device = next(decoder.parameters()).device
    optimiser = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=patience, min_lr=1e-6
    )
    num_frames = loader.dataset.tensors[1].shape[-1]
    best_loss = np.inf

    with progress() as bar:
        epoch_task = bar.add_task("Epochs", total=epochs)
        step_task = bar.add_task("Steps", total=len(loader), visible=False)

        for epoch in range(epochs):
            bar.update(step_task, completed=0, visible=True)
            decoder.train()
            running_loss = 0.0

            for embeddings, targets in loader:
                embeddings, targets = embeddings.to(device), targets.to(device)
                reconstruction = decoder(embeddings, num_frames)
                frames = min(reconstruction.shape[-1], targets.shape[-1])
                loss = torch.nn.functional.mse_loss(reconstruction[..., :frames], targets[..., :frames])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
                bar.advance(step_task)

            epoch_loss = running_loss / len(loader)
            scheduler.step(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                ensure_dir(DECODER_CHECKPOINT.parent)
                torch.save(decoder.state_dict(), DECODER_CHECKPOINT)

            bar.update(epoch_task, advance=1, description=f"Epoch {epoch + 1}/{epochs} · MSE {epoch_loss:.8f}")
            bar.update(step_task, visible=False)

    return float(best_loss)

@torch.no_grad()
def reconstruct(decoder: InceptionDecoder, loader: DataLoader, num_frames: int) -> np.ndarray:
    """Decode every embedding back into a pitch contour and a voicing mask."""
    device = next(decoder.parameters()).device
    decoder.eval()
    return np.concatenate([decoder(embeddings.to(device), num_frames).cpu().numpy() for embeddings, *_ in loader])

def evaluate(reference: torch.Tensor, reconstructions: np.ndarray) -> None:
    """Score every reconstruction and write the metric table."""
    pitch = denormalise_cents(np.clip(reconstructions[:, 0], 0, 1))
    voicing = reconstructions[:, 1] > VOICING_THRESHOLD
    scores = ReconstructionScores()

    with progress() as bar:
        task = bar.add_task("Evaluating reconstructions", total=len(reference))
        for index in range(len(reference)):
            original = reference[index].numpy()
            length = int(np.count_nonzero(~np.isnan(original)))

            truth = denormalise_cents(original[:length])
            truth[np.isnan(original[:length])] = np.nan

            estimate = pitch[index][:length].copy()
            estimate[~voicing[index][:length]] = np.nan

            scores.add(truth, estimate)
            bar.advance(task)

    means = scores.means()
    ensure_dir(RESULTS_DIR)
    pd.DataFrame([means]).to_csv(RESULTS_DIR / RESULTS_FILE, sep="\t", index=False)

    metrics_table(
        "Svara synthesis · reconstruction error",
        ["Metric", "Mean"],
        [
            ("DTW distance", f"{means['dtw_distance_cents']:.2f} cents"),
            ("Periodicity error", f"{means['periodicity_error_oscillations']:.2f} oscillations"),
            ("Pitch position error", f"{means['pitch_position_error_cents']:.2f} cents"),
        ],
    )
