"""
Tetrahedral Transformer Layer
Wraps TetrahedralAttention + FFN + LayerNorm into a full transformer encoder layer.
Drop-in replacement for nn.TransformerEncoderLayer with geometric bias support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .tetrahedral_attention import TetrahedralAttention


class TetrahedralTransformerLayer(nn.Module):
    """Transformer encoder layer with TetrahedralAttention for self-attention.

    Architecture (pre-norm):
        x → LayerNorm → TetrahedralAttention → residual → LayerNorm → FFN → residual
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.1,
        use_geometric_bias: bool = True,
    ):
        super().__init__()
        self.self_attn = TetrahedralAttention(
            hidden_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_geometric_bias=use_geometric_bias,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        geometric_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            geometric_bias: [seq_len, seq_len] optional geometric bias
            attention_mask: [batch, seq_len] optional mask (0=pad)
        Returns:
            [batch, seq_len, d_model]
        """
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(
            normed,
            geometric_bias=geometric_bias,
            attention_mask=attention_mask,
        )
        x = x + self.dropout(attn_out)

        # Pre-norm FFN
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out

        return x


class TetrahedralTransformerEncoder(nn.Module):
    """Stack of TetrahedralTransformerLayers with shared geometric bias generation."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        dropout: float = 0.1,
        use_geometric_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            TetrahedralTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_ff=dim_ff,
                dropout=dropout,
                use_geometric_bias=use_geometric_bias,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Learnable position-dependent geometric bias seed
        # The bias is generated from this via a lightweight projection
        if use_geometric_bias:
            self.geo_proj = nn.Linear(d_model, 1, bias=False)
        else:
            self.geo_proj = None

    def _make_geometric_bias(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """Generate a [seq_len, seq_len] geometric bias from position embeddings.

        The bias encodes relative positional proximity: closer positions get
        stronger attention bias. This is the core tetrahedral geometry prior.
        """
        if self.geo_proj is None:
            return None

        # Position indices
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Relative distance matrix: |i - j|
        rel_dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # Gaussian decay: bias = exp(-dist^2 / sigma^2)
        # sigma scales with sqrt(d_model) for larger models
        sigma = math.sqrt(self.d_model)
        bias = torch.exp(-rel_dist.pow(2) / (2 * sigma ** 2))
        # Normalize to [-1, 1] range
        bias = 2.0 * (bias - bias.min()) / (bias.max() - bias.min() + 1e-8) - 1.0
        return bias

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] optional (0=pad)
        Returns:
            [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        geometric_bias = self._make_geometric_bias(seq_len, x.device)

        for layer in self.layers:
            x = layer(x, geometric_bias=geometric_bias, attention_mask=attention_mask)

        return self.final_norm(x)


if __name__ == "__main__":
    import torch
    B, S, D, H, L = 2, 32, 256, 8, 4
    enc = TetrahedralTransformerEncoder(D, H, L, D * 4)
    x = torch.randn(B, S, D)
    out = enc(x)
    print(f"Input: {x.shape}  Output: {out.shape}")
    params = sum(p.numel() for p in enc.parameters())
    print(f"Params: {params/1e6:.1f}M")
    print("OK")
