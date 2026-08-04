"""
Shared masking utilities.

A single, reusable `masked_mean` helper for pooling over the sequence
dimension while correctly excluding padding positions.

Background: several modules in this codebase pool token-level features
into a single per-example vector via a naive `x.mean(dim=1)`. Training
batches are right-padded to the batch's longest example, so a naive mean
is diluted by however much padding happens to be present -- which varies
batch-to-batch. Real single-sequence autoregressive generation never has
padding, so any pooled feature computed this way is systematically
out-of-distribution at inference time relative to what the model saw
during training. Using `attention_mask` to exclude padding from the mean
keeps the pooled feature consistent between training and generation.
"""

from typing import Optional

import torch


def masked_mean(
    x: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dim: int = 1,
) -> torch.Tensor:
    """
    Mean-pool `x` over `dim` (the sequence dimension), excluding padding
    positions when `attention_mask` is given.

    Args:
        x: Tensor of shape [..., seq_len, ...] with the sequence axis at `dim`.
        attention_mask: Optional binary mask [batch, seq_len] (1=real, 0=pad),
            broadcastable against `x` once unsqueezed on the last dim. If
            `None`, falls back to a plain `x.mean(dim=dim)`.
        dim: Sequence dimension to reduce over (default 1, i.e. [B, S, D]).

    Returns:
        Tensor with `dim` reduced out, e.g. [batch, hidden_dim].
    """
    if attention_mask is None:
        return x.mean(dim=dim)
    mask = attention_mask.to(x.dtype).unsqueeze(-1)  # [B, S, 1]
    summed = (x * mask).sum(dim=dim)
    count = mask.sum(dim=dim).clamp(min=1e-8)
    return summed / count
