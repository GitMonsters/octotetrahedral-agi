"""
Ethics Limb — Value alignment and safety constraint enforcement

Acts as hard boundary constraints on the output manifold.
When activated, contracts the allowable output space (safety contraction).
Not a filter on outputs — operates in latent space to geometrically
prevent unsafe directions from being representable.

Implementation:
- Safety basis vectors: learned unsafe directions in latent space
- Constraint projection: removes unsafe components from hidden states
- Boundary detector: flags when representations approach unsafe regions
- Value alignment score: measures how well outputs align with values
- Contraction/expansion: dynamically adjusts allowable output region
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .base_limb import BaseLimb


class SafetyBasisProjector(nn.Module):
    """
    Learns a set of "unsafe directions" in latent space.
    Projects hidden states to remove unsafe components.
    Geometric safety contraction.
    """

    def __init__(self, hidden_dim: int, num_safety_axes: int = 8):
        super().__init__()
        self.num_safety_axes = num_safety_axes
        # Learnable unsafe directions (orthonormalized during forward)
        self.unsafe_bases = nn.Parameter(torch.randn(num_safety_axes, hidden_dim) * 0.01)
        # Threshold: how close to unsafe before contracting
        self.threshold_head = nn.Sequential(
            nn.Linear(hidden_dim, num_safety_axes),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [B, S, D]
        Returns:
            (safe_hidden, safety_scores)
        """
        # Orthonormalize safety bases (Gram-Schmidt approximation)
        bases = F.normalize(self.unsafe_bases, dim=-1)  # [A, D]

        # Compute projection onto each unsafe axis
        # [B, S, D] @ [A, D]^T → [B, S, A]
        projections = torch.einsum('bsd,ad->bsa', hidden, bases)

        # Safety score per axis: how much is aligned with unsafe direction
        safety_scores = projections.abs().mean(dim=1)  # [B, A]

        # Determine contraction strength per axis
        pooled = hidden.mean(dim=1)  # [B, D]
        thresholds = self.threshold_head(pooled)  # [B, A]

        # Remove unsafe components where projection exceeds threshold
        # Soft removal: scale down unsafe components
        mask = torch.sigmoid(10 * (projections.abs() - thresholds.unsqueeze(1)))  # [B, S, A]
        removal = torch.einsum('bsa,ad->bsd', projections * mask, bases)
        safe_hidden = hidden - removal

        return safe_hidden, safety_scores


class ValueAligner(nn.Module):
    """Measures and enforces alignment between outputs and learned values."""

    def __init__(self, hidden_dim: int, num_values: int = 4):
        super().__init__()
        # Learnable value direction vectors
        self.value_vectors = nn.Parameter(torch.randn(num_values, hidden_dim) * 0.01)
        # Value names (semantic — not used in computation)
        self.num_values = num_values

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (alignment_scores, alignment_loss)
        """
        values = F.normalize(self.value_vectors, dim=-1)  # [V, D]
        pooled = F.normalize(hidden.mean(dim=1), dim=-1)  # [B, D]

        # Cosine alignment with each value
        alignment = torch.mm(pooled, values.t())  # [B, V]

        # Loss: encourage positive alignment (outputs should align with values)
        alignment_loss = F.relu(-alignment).mean()  # Penalize anti-alignment

        return alignment, alignment_loss


class EthicsLimb(BaseLimb):
    """
    Enforces value alignment and safety constraints in latent space.

    Acts as geometric safety contraction:
    - Detects proximity to unsafe directions
    - Removes unsafe vector components
    - Measures value alignment
    - Dynamically contracts/expands allowable region
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_safety_axes: int = 8,
        num_values: int = 4,
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
            limb_name="ethics",
        )

        self.safety_projector = SafetyBasisProjector(hidden_dim, num_safety_axes)
        self.value_aligner = ValueAligner(hidden_dim, num_values)

        # Ethics processing
        self.ethics_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Boundary alarm
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self._safety_scores = None
        self._value_alignment = None
        self._alignment_loss = torch.tensor(0.0)

    def process(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Safety projection: remove unsafe components
        safe_x, safety_scores = self.safety_projector(x)
        self._safety_scores = safety_scores.detach()

        # Value alignment
        alignment, alignment_loss = self.value_aligner(safe_x)
        self._value_alignment = alignment.detach()
        self._alignment_loss = alignment_loss

        # Ethics-aware processing
        refined = self.ethics_mlp(safe_x)
        return self.norm(safe_x + refined)

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
            # Confidence = how safely within bounds
            confidence = (1.0 - self.boundary_head(pooled)).mean().item()

        return output, confidence, None

    def get_alignment_loss(self) -> torch.Tensor:
        """Value alignment loss for training."""
        return self._alignment_loss

    def get_safety_state(self) -> Dict[str, Any]:
        return {
            'safety_scores': (
                self._safety_scores.mean(dim=0).tolist()
                if self._safety_scores is not None else []
            ),
            'value_alignment': (
                self._value_alignment.mean(dim=0).tolist()
                if self._value_alignment is not None else []
            ),
            'alignment_loss': self._alignment_loss.item(),
        }

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update(self.get_safety_state())
        return stats
