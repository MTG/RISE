"""The three models built on the InceptionTime backbone.

Each corresponds to one stage of the thesis and exposes exactly the interface that
stage needs, so that no model carries branches for a task it is not performing:

``ContrastiveModel``
    encoder plus projection head, trained with InfoNCE during pretraining.
``SvaraEmbedder``
    frozen encoder plus pooling, used for melodic pattern retrieval.
``ContextualSvaraClassifier``
    three encoders reading the preceding, current and succeeding contours, joined
    by co-attention; used for *svara* classification and *svara*-form clustering.

All three hold the backbone under the attribute ``encoder``, so a checkpoint
written as ``model.encoder.state_dict()`` is loadable by any of them.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import CoAttention
from .inception import InceptionEncoder
from .lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK, apply_lora

#: Pitch plus its binary silence mask.
NUM_INPUT_FEATURES = 2

#: The preceding, current and succeeding streams of the contextual classifier.
CONTEXT_STREAMS = ("prec", "curr", "succ")


def global_average_pool(features: torch.Tensor) -> torch.Tensor:
    """Collapse ``(batch, channels, frames)`` to ``(batch, channels)`` over time.

    Averaging over time is what makes the embedding independent of how long the
    *svara* was held, so that two renditions of the same *svara* at different
    tempi land in the same region of the space.
    """
    return features.mean(dim=-1)


class ProjectionHead(nn.Sequential):
    """The SimCLR projection ``G(h)``, discarded after pretraining."""

    def __init__(self, embed_dim: int, out_dim: int) -> None:
        super().__init__(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )


class ContrastiveModel(nn.Module):
    """Encoder ``F(x)`` and projection ``G(h)`` optimised with the InfoNCE loss."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        out_dim: int,
        num_features: int = NUM_INPUT_FEATURES,
    ) -> None:
        super().__init__()
        self.encoder = InceptionEncoder(embed_dim, depth, num_features)
        self.projection = ProjectionHead(embed_dim, out_dim)

    def forward(self, contours: torch.Tensor) -> torch.Tensor:
        """Return the ``(batch, out_dim)`` projections ``z`` for a batch of contours."""
        return self.projection(global_average_pool(self.encoder(contours)))


class SvaraEmbedder(nn.Module):
    """Pooled encoder embeddings, L2-normalised so that cosine similarity is a dot product."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_features: int = NUM_INPUT_FEATURES,
        *,
        normalise: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = InceptionEncoder(embed_dim, depth, num_features)
        self.normalise = normalise

    def forward(self, contours: torch.Tensor) -> torch.Tensor:
        """Return the ``(batch, embed_dim)`` embedding of each contour."""
        embeddings = global_average_pool(self.encoder(contours))
        if self.normalise:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class ContextualSvaraClassifier(nn.Module):
    """Classify a *svara* from its own contour and those of its two neighbours.

    Each stream has its own encoder and its own GRU; co-attention then summarises
    all three with respect to the final hidden state of the current stream, and the
    concatenation of the three context vectors is classified. Attaching the
    attention to the current stream is what makes the context *relative*: the model
    looks at the neighbours for whatever bears on the *svara* in hand.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_classes: int,
        num_features: int = NUM_INPUT_FEATURES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.prec_encoder = InceptionEncoder(embed_dim, depth, num_features)
        self.curr_encoder = InceptionEncoder(embed_dim, depth, num_features)
        self.succ_encoder = InceptionEncoder(embed_dim, depth, num_features)

        self.prec_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.curr_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.succ_gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

        self.attention = CoAttention(embed_dim)
        self.head = nn.Sequential(
            nn.BatchNorm1d(embed_dim * len(CONTEXT_STREAMS)),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * len(CONTEXT_STREAMS), embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    @property
    def encoder(self) -> InceptionEncoder:
        """The current-stream encoder, i.e. the one that sees the *svara* itself."""
        return self.curr_encoder

    @property
    def encoders(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return self.prec_encoder, self.curr_encoder, self.succ_encoder

    def forward(self, prec: torch.Tensor, curr: torch.Tensor, succ: torch.Tensor) -> torch.Tensor:
        """Return the ``(batch, num_classes)`` logits."""
        return self.head(self.encode(prec, curr, succ))

    def encode(self, prec: torch.Tensor, curr: torch.Tensor, succ: torch.Tensor) -> torch.Tensor:
        """Return the ``(batch, 3 * embed_dim)`` representation the head classifies.

        This is the vector clustered in the *svara*-form experiment: the head is
        discarded, and the geometry of this space is what is evaluated.
        """
        prec_states = self._run(prec, self.prec_encoder, self.prec_gru)
        curr_states = self._run(curr, self.curr_encoder, self.curr_gru)
        succ_states = self._run(succ, self.succ_encoder, self.succ_gru)
        reference = curr_states[:, -1:]

        return torch.cat(
            [
                self.attention(prec_states, reference),
                self.attention(curr_states, reference),
                self.attention(succ_states, reference),
            ],
            dim=1,
        )

    @staticmethod
    def _run(contours: torch.Tensor, encoder: nn.Module, gru: nn.GRU) -> torch.Tensor:
        hidden_states, _ = gru(encoder(contours).permute(0, 2, 1))
        return hidden_states

    def apply_lora(
        self,
        rank: int = DEFAULT_RANK,
        alpha: int = DEFAULT_ALPHA,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Replace the three encoders in place with LoRA-adapted versions."""
        self.prec_encoder = apply_lora(self.prec_encoder, rank, alpha, dropout)
        self.curr_encoder = apply_lora(self.curr_encoder, rank, alpha, dropout)
        self.succ_encoder = apply_lora(self.succ_encoder, rank, alpha, dropout)

    def load_pretrained_encoders(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialise all three encoders from one pretrained backbone checkpoint."""
        for encoder in self.encoders:
            encoder.load_state_dict(state_dict)

    def set_encoders_trainable(self, trainable: bool = True) -> None:
        """Freeze or unfreeze the three encoders, leaving the head unaffected."""
        for encoder in self.encoders:
            for parameter in encoder.parameters():
                parameter.requires_grad = trainable

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
