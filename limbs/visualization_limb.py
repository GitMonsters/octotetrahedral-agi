"""
Visualization Limb — Reconstructive mental image formation

Key properties (per user's theory):
- Requires prior knowledge (memory retrieval)
- Multi-resolution inspection: zoom in/out of mental images
- Detail-oriented: captures fine structure
- Tangible: works with concrete representations
- Reconstructive: past → present

Implementation:
- Reads from memory to reconstruct representations at multiple scales
- Multi-scale attention (coarse → fine) mimics zooming
- Detail refinement layers progressively sharpen the image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .base_limb import BaseLimb


class MultiScaleAttention(nn.Module):
    """Attend at multiple resolutions — coarse global then fine local."""

    def __init__(self, hidden_dim: int, num_scales: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_scales = num_scales
        self.scale_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=max(1, 4 // (i + 1)),
                                  dropout=dropout, batch_first=True)
            for i in range(num_scales)
        ])
        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_scales)
        ])
        # Learnable zoom weights
        self.zoom_gate = nn.Linear(hidden_dim, num_scales)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        kv = memory if memory is not None else x

        # Determine zoom weights from context
        zoom_weights = torch.softmax(self.zoom_gate(x.mean(dim=1)), dim=-1)  # [B, scales]

        out = torch.zeros_like(x)
        for i, (attn, norm) in enumerate(zip(self.scale_attns, self.scale_norms)):
            # Progressively downsample for coarser scales
            stride = 2 ** i
            if stride > 1 and S > stride:
                kv_scaled = kv[:, ::stride, :]
            else:
                kv_scaled = kv
            attn_out, _ = attn(x, kv_scaled, kv_scaled)
            attn_out = norm(attn_out)
            out = out + zoom_weights[:, i].unsqueeze(-1).unsqueeze(-1) * attn_out

        return out


class VisualizationLimb(BaseLimb):
    """
    Forms detailed mental images from memory at multiple resolutions.

    Unlike imagination, visualization is:
    - Reconstructive (needs prior knowledge)
    - Detail-focused (zoom in/out)
    - Tangible (concrete representations)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_scales: int = 3,
        dropout: float = 0.1,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        buffer_size: int = 100,
    ):
        super().__init__(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            buffer_size=buffer_size,
            limb_name="visualization",
        )

        self.multi_scale = MultiScaleAttention(hidden_dim, num_scales, dropout)

        # Detail refinement: progressively sharpen
        self.detail_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.detail_norm = nn.LayerNorm(hidden_dim)

        # Resolution score: how detailed is this visualization?
        self.resolution_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def process(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Multi-scale inspection (zoom in/out)
        visualized = self.multi_scale(x)
        # Detail refinement
        refined = self.detail_refiner(visualized)
        output = self.detail_norm(visualized + refined)
        return output

    def forward(
        self, x: torch.Tensor, return_confidence: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor]]:
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        output = self.process(adapted, **kwargs)

        confidence = None
        if return_confidence:
            pooled = output.mean(dim=1)
            confidence = self.resolution_head(pooled).mean().item()

        return output, confidence, None
