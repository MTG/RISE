"""Training loops shared by the experiments.

The contextual classifier is trained identically for *svara* classification and for
*svara*-form clustering — only the label alphabet differs — so the loop lives here
rather than in either experiment. Checkpointing and resumption live here too, so
that a long fine-tuning run can be interrupted and picked up again.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .console import detail, progress, warn
from .data.datasets import append_silence_mask
from .paths import CACHE_DIR, ensure_dir


@dataclass
class TrainingState:
    """What has to survive an interruption for a run to resume where it stopped.

    The counters are not enough on their own. Restoring only ``epoch`` would drop a
    freshly initialised model into the middle of its budget, carrying a best score it
    has not earned and a patience it has already spent, so the live weights and the
    optimiser moments belong here too. Those are two orders of magnitude larger than
    the counters — a few hundred megabytes against a few bytes — so they are written
    only when a run is actually interrupted, which is the case resumption exists for;
    the counters are written at the end of every epoch.
    """

    epoch: int = 0
    best_score: float = -np.inf
    epochs_without_improvement: int = 0
    stopped: bool = False
    model: dict[str, object] | None = None
    optimiser: dict[str, object] | None = None

    @classmethod
    def load(cls, path: Path) -> TrainingState:
        if not path.exists():
            return cls()
        try:
            stored = torch.load(path, map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError, OSError):
            warn(f"Ignoring a corrupt resume record at {path}")
            return cls()
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in stored.items() if key in known})

    @property
    def resumable(self) -> bool:
        """Whether the live state was captured and not merely the counters."""
        return self.model is not None and self.optimiser is not None

    def capture(self, model: nn.Module, optimiser: torch.optim.Optimizer) -> None:
        """Take the live weights and optimiser moments into the state."""
        self.model = model.state_dict()
        self.optimiser = optimiser.state_dict()

    def forget_weights(self) -> None:
        """Drop weights that an epoch of training has made stale."""
        self.model = self.optimiser = None

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        staged = path.with_suffix(".tmp")
        torch.save(vars(self), staged)
        staged.replace(path)


def run_dir(tag: str) -> Path:
    """Where the intermediates of the run named ``tag`` are kept.

    Under the cache, not under ``checkpoints/``: everything a downstream run writes
    is regenerable from the corpora and the two published models, and only those two
    are worth keeping. Deleting the cache costs compute, never provenance.
    """
    return CACHE_DIR / "runs" / tag


def resume_path(tag: str) -> Path:
    """The state that lets an interrupted run pick up where it stopped."""
    return run_dir(tag) / "resume.pt"


def best_weights_path(tag: str) -> Path:
    """The best-scoring weights of the run, reloaded to score the held-out split."""
    return run_dir(tag) / "best.pth"


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return the true and predicted labels over ``loader``, in evaluation mode."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    true, predicted = [], []
    for prec, curr, succ, targets in loader:
        logits = model(
            append_silence_mask(prec.to(device)),
            append_silence_mask(curr.to(device)),
            append_silence_mask(succ.to(device)),
        )
        predicted.append(logits.argmax(dim=1).cpu().numpy())
        true.append(targets.numpy())

    model.train(was_training)
    return np.concatenate(true), np.concatenate(predicted)


def macro_f1(model: nn.Module, loader: DataLoader) -> float:
    """Macro-averaged F1 over ``loader``.

    Macro rather than micro because the *svaras* of a *rāga* are far from equally
    frequent — a *rāga* dwells on some degrees and passes through others — and a
    micro average would let the common ones hide failure on the rare ones.
    """
    true, predicted = predict(model, loader)
    return float(f1_score(true, predicted, average="macro"))


@torch.no_grad()
def encode(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """Stack the contextual representations of every observation in ``loader``."""
    device = next(model.parameters()).device
    model.eval()
    embeddings = [
        model.encode(
            append_silence_mask(prec.to(device)),
            append_silence_mask(curr.to(device)),
            append_silence_mask(succ.to(device)),
        )
        .cpu()
        .numpy()
        for prec, curr, succ, *_ in loader
    ]
    return np.concatenate(embeddings, axis=0)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    head_warmup_epochs: int,
    run_tag: str,
) -> float:
    """Fine-tune the contextual classifier, returning the best validation F1.

    For the first ``head_warmup_epochs`` the encoders are frozen so that the
    randomly initialised head can settle before it starts sending gradients into a
    pretrained encoder; unfreezing earlier would let the noise of an untrained head
    undo the representation that pretraining produced.
    """
    device = next(model.parameters()).device
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    checkpoint = best_weights_path(run_tag)

    state = TrainingState.load(resume_path(run_tag))
    if state.epoch and not state.resumable:
        warn(f"{run_tag}: the interrupted weights were not saved; restarting this run")
        state = TrainingState()
    if state.stopped or state.epoch >= epochs:
        detail(f"Already trained for {state.epoch} epochs; best validation F1 {state.best_score:.4f}")
        return float(state.best_score)
    if state.epoch:
        model.load_state_dict(state.model)
        optimiser.load_state_dict(state.optimiser)
        detail(f"Resuming at epoch {state.epoch} with best validation F1 {state.best_score:.4f}")

    try:
        with progress() as bar:
            epoch_task = bar.add_task("Epochs", total=epochs, completed=state.epoch)
            step_task = bar.add_task("Steps", total=len(train_loader), visible=False)

            for epoch in range(state.epoch, epochs):
                bar.update(step_task, completed=0, visible=True)
                model.set_encoders_trainable(epoch >= head_warmup_epochs)

                train_f1 = macro_f1(model, train_loader)
                val_f1 = macro_f1(model, val_loader)

                model.train()
                for prec, curr, succ, targets in train_loader:
                    logits = model(
                        append_silence_mask(prec.to(device)),
                        append_silence_mask(curr.to(device)),
                        append_silence_mask(succ.to(device)),
                    )
                    loss = nn.functional.cross_entropy(logits, targets.to(device))
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    bar.advance(step_task)

                bar.update(
                    epoch_task,
                    advance=1,
                    description=f"Epoch {epoch + 1}/{epochs} · train F1 {train_f1:.4f} · val F1 {val_f1:.4f}",
                )
                bar.update(step_task, visible=False)

                if val_f1 > state.best_score:
                    state.best_score = val_f1
                    state.epochs_without_improvement = 0
                    save_checkpoint(model, checkpoint)
                else:
                    state.epochs_without_improvement += 1

                state.epoch = epoch + 1
                state.stopped = state.epochs_without_improvement >= patience
                state.forget_weights()
                state.save(resume_path(run_tag))

                if state.stopped:
                    detail(f"Early stopping after {patience} epochs without improvement")
                    break
    except KeyboardInterrupt:
        state.capture(model, optimiser)
        state.save(resume_path(run_tag))
        detail(f"Saved the training state of {run_tag} at epoch {state.epoch}")
        raise

    return float(state.best_score)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(path, map_location=device))
    return model
