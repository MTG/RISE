"""Low-Rank Adaptation of the pretrained encoders.

Fine-tuning the full encoder on a few hundred annotated *svaras* of a single
*rāga* overfits it and discards the *rāga*-independence that pretraining bought.
LoRA instead freezes the pretrained weights and learns a rank-``r`` update
``dW = (alpha / r) * B @ A`` beside them, which is small enough to be estimated
from the annotations that exist.
"""

from __future__ import annotations

from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

#: Rank of the low-rank update.
DEFAULT_RANK = 8

#: Scaling factor applied to the update.
DEFAULT_ALPHA = 16

DEFAULT_DROPOUT = 0.05


def apply_lora(
    module: nn.Module,
    rank: int = DEFAULT_RANK,
    alpha: int = DEFAULT_ALPHA,
    dropout: float = DEFAULT_DROPOUT,
) -> PeftModel:
    """Wrap every 1-D convolution in ``module`` with a LoRA adapter."""
    targets = [name for name, child in module.named_modules() if isinstance(child, nn.Conv1d)]
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=targets)
    return get_peft_model(module, config)
