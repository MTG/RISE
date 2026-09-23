"""Co-attention over the preceding, current and succeeding encodings.

Which form a *svara* takes depends on the *svaras* around it, but not every frame
of that surrounding context matters equally. Attention is therefore computed with
respect to the final hidden state of the *current* encoder, so that each of the
three streams is summarised by whatever part of it is relevant to the *svara*
being classified, rather than by a fixed pooling.
"""

from __future__ import annotations

import torch
from torch import nn


class CoAttention(nn.Module):
    """Attend over a hidden-state sequence with a learned bilinear compatibility.

    For hidden states ``H`` of shape ``(batch, frames, embed_dim)`` and a reference
    state ``h_c``, the compatibility of frame ``t`` is ``e_t = h_c^T W H_t``. The
    scores are normalised over time and used to take a weighted sum of ``H``:

    .. math::
        \\alpha_t = \\frac{\\exp(e_t)}{\\sum_k \\exp(e_k)},
        \\qquad c = \\sum_t \\alpha_t H_t.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Summarise ``hidden_states`` with respect to ``reference``.

        ``hidden_states`` is ``(batch, frames, embed_dim)`` and ``reference`` is the
        ``(batch, 1, embed_dim)`` final hidden state of the current stream. Returns
        the ``(batch, embed_dim)`` context vector.
        """
        scores = torch.bmm(self.project(hidden_states), reference.permute(0, 2, 1))
        weights = scores.softmax(dim=1)
        return torch.bmm(hidden_states.permute(0, 2, 1), weights).squeeze(-1)
