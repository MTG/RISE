"""InceptionTime encoder and its transposed counterpart, the synthesis decoder.

The encoder follows InceptionTime (Ismail Fawaz et al., 2020): a stack of blocks,
each of which convolves its input with three kernels of different width in
parallel and adds a bottleneck residual. The three widths give one block a view of
a *svara* at three time scales at once — roughly a single inflection, a lobe of an
oscillation, and a whole ornament — which is what lets a fixed-depth stack cope
with contours whose ornamentation varies from slow to very fast.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn

#: Receptive field of the three parallel branches, in frames.
KERNEL_SIZES = (9, 19, 39)

#: Number of parallel convolutional branches per block.
NUM_BRANCHES = len(KERNEL_SIZES)

#: Branch attribute names used before the branches became a :class:`~torch.nn.ModuleList`.
#: Checkpoints published with the paper still carry them, so they are remapped on load.
LEGACY_BRANCH_PREFIXES = tuple(f"branch{index + 1}." for index in range(NUM_BRANCHES))


def branch_widths(out_channels: int) -> list[int]:
    """Split ``out_channels`` across the branches, giving the remainder to the first.

    The three branch outputs are concatenated and added to a residual of width
    ``out_channels``, so the widths have to sum to exactly ``out_channels``. Any
    ``out_channels`` divisible by :data:`NUM_BRANCHES` — as every configuration in
    ``configs.yaml`` is — splits evenly and the remainder term vanishes.
    """
    base, remainder = divmod(out_channels, NUM_BRANCHES)
    return [base + (1 if index < remainder else 0) for index in range(NUM_BRANCHES)]


class _LegacyBranchNames:
    """Accept ``branch1``/``branch2``/``branch3`` keys from pre-ModuleList checkpoints."""

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for index, legacy in enumerate(LEGACY_BRANCH_PREFIXES):
            for key in [k for k in state_dict if k.startswith(prefix + legacy)]:
                renamed = key.replace(prefix + legacy, f"{prefix}branches.{index}.", 1)
                state_dict[renamed] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class InceptionBlock(_LegacyBranchNames, nn.Module):
    """Three parallel convolutions concatenated, plus a 1x1 residual projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(in_channels, width, kernel_size=size, padding=size // 2)
            for size, width in zip(KERNEL_SIZES, branch_widths(out_channels), strict=True)
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(inputs) for branch in self.branches], dim=1) + self.residual(inputs)


class TransposeInceptionBlock(_LegacyBranchNames, nn.Module):
    """:class:`InceptionBlock` with transposed convolutions, for upsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.ConvTranspose1d(in_channels, width, kernel_size=size, padding=size // 2)
            for size, width in zip(KERNEL_SIZES, branch_widths(out_channels), strict=True)
        )
        self.residual = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(inputs) for branch in self.branches], dim=1) + self.residual(inputs)


class InceptionEncoder(nn.Module):
    """Encode a pitch contour into a ``(batch, embed_dim, frames)`` representation.

    Channel width starts at ``embed_dim * 2 ** (depth - 1)`` and halves at every
    block while average pooling halves the temporal resolution, so the stack trades
    time for channels at a constant rate. The final block omits the pooling, and
    leaves the time axis intact for the downstream model to consume — pooled to a
    single vector for contrastive learning and retrieval, or read step by step by
    the recurrent layers of the contextual classifier.
    """

    def __init__(self, embed_dim: int, depth: int, num_features: int = 2) -> None:
        super().__init__()
        widths = [embed_dim * 2**exponent for exponent in range(depth - 1, 0, -1)]
        channels = [num_features, *widths, embed_dim]

        blocks = [
            self._pooled_block(in_channels, out_channels)
            for in_channels, out_channels in pairwise(channels[:-1])
        ]
        blocks.append(InceptionBlock(channels[-2], channels[-1]))
        self.blocks = nn.ModuleList(blocks)

    @staticmethod
    def _pooled_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            InceptionBlock(in_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

    def forward(self, contours: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            contours = block(contours)
        return contours


class InceptionDecoder(nn.Module):
    """Reconstruct a pitch contour and its silence mask from an encoder embedding.

    The block stack mirrors :class:`InceptionEncoder` in reverse — channels halve
    into time — and the two heads separate the two things a *svara* contour has to
    say: where the pitch is, and whether there is any pitch at all. Predicting the
    mask separately keeps unvoiced frames from dragging the regression head
    towards a pitch value they do not have.
    """

    def __init__(self, embed_dim: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList(
            nn.Sequential(
                TransposeInceptionBlock(embed_dim * 2**exponent, embed_dim * 2 ** (exponent + 1)),
                nn.BatchNorm1d(embed_dim * 2 ** (exponent + 1)),
                nn.ReLU(),
            )
            for exponent in range(depth - 1)
        )

        width = embed_dim * 2 ** (depth - 1)
        self.pitch_head = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 1),
        )
        self.mask_head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Return ``(batch, 2, num_frames)``: normalised pitch, then silence mask."""
        for step, block in enumerate(self.blocks):
            embeddings = block(embeddings)
            size = num_frames // 2 ** (self.depth - 2 - step)
            embeddings = nn.functional.interpolate(embeddings, size=size, mode="linear", align_corners=False)

        frames = embeddings.permute(0, 2, 1)
        pitch = self.pitch_head(frames)
        mask = torch.sigmoid(self.mask_head(frames))
        return torch.cat([pitch, mask], dim=-1).permute(0, 2, 1)
