"""
Rotary Position Embedding (RoPE)

Encodes token position by rotating pairs of query/key dimensions by an
angle proportional to (position * frequency), rather than adding a fixed
positional vector to the token embedding.

Why RoPE over additive sinusoidal/absolute encoding:
- Position information is injected directly into the attention dot-product
  (Q . K), so attention scores naturally become a function of *relative*
  position (i - j), not absolute position. This generalizes better to
  positions/lengths seen less often during training.
- No separate learned or fixed positional vector competes with the token
  embedding itself.

Reference formulation (as in GPT-NeoX / LLaMA):
    theta_i = base ** (-2i / dim),  i = 0 .. dim/2 - 1
    for position p: rotate each (x_i, x_{i+dim/2}) pair by angle p * theta_i
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Precomputes and caches cos/sin rotation tables for RoPE, extending the
    cache lazily if a longer sequence is seen than initially provisioned.
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._cached_seq_len = 0
        # Non-persistent: these are deterministic functions of (dim, base) and
        # don't need to be saved/loaded with the model checkpoint.
        self.register_buffer('cos_cached', torch.zeros(0, dim), persistent=False)
        self.register_buffer('sin_cached', torch.zeros(0, dim), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self._cached_seq_len = seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) tables of shape [seq_len, dim], extending the cache if needed."""
        if seq_len > self._cached_seq_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Splits the last dim in half and rotates: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embedding to query/key tensors.

    Args:
        q, k: [batch, num_heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim]

    Returns:
        Rotated (q, k), same shapes as input.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


if __name__ == "__main__":
    # Smoke test
    batch, heads, seq_len, head_dim = 2, 8, 16, 32
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=64)
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    cos, sin = rope(seq_len, device=q.device, dtype=q.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    # Rotation must preserve vector norm per position/head.
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)

    # Cache should extend automatically for longer sequences.
    cos2, sin2 = rope(128, device=q.device, dtype=q.dtype)
    assert cos2.shape == (128, head_dim)
    print("RotaryEmbedding smoke test passed.")
