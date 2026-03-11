"""
Imagination Limb — Generative exploration beyond known latent space

Key properties (per user's theory):
- Does NOT require prior knowledge
- Cross-modal: "feel a hug", "smell dew", "taste spicy chocolate"
- Experience-first, not detail-first
- Intangible + tangible
- Generative: present → possible (explores unmapped territory)
- "In the moment" — experiential, not analytical

Implementation:
- Noise injection for novelty (explore beyond training manifold)
- Cross-modal blending (mix representation channels)
- Experience synthesis via variational sampling
- Distinct from visualization: generates new, doesn't reconstruct old
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .base_limb import BaseLimb


class NoveltyExplorer(nn.Module):
    """Injects controlled noise to push beyond the training manifold."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Learnable novelty magnitude
        self.novelty_scale = nn.Parameter(torch.tensor(0.1))
        # Novelty direction is data-dependent
        self.direction_net = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direction = F.normalize(self.direction_net(x), dim=-1)
        noise = torch.randn_like(x) * self.novelty_scale.abs()
        # Project noise along learned novel directions
        return x + direction * noise


class CrossModalBlender(nn.Module):
    """
    Blends representation channels to simulate cross-modal experience.
    "What would X feel/smell/taste like?" requires mixing modality axes.
    """

    def __init__(self, hidden_dim: int, num_modalities: int = 4):
        super().__init__()
        self.num_modalities = num_modalities
        chunk_dim = hidden_dim // num_modalities

        # Each "modality" is a chunk of the hidden dimension
        self.modality_transforms = nn.ModuleList([
            nn.Linear(chunk_dim, chunk_dim) for _ in range(num_modalities)
        ])
        # Cross-modal mixing matrix (learnable)
        self.mix_matrix = nn.Parameter(
            torch.eye(num_modalities) + 0.1 * torch.randn(num_modalities, num_modalities)
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.chunk_dim = chunk_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # Split into modality chunks
        chunks = x.split(self.chunk_dim, dim=-1)
        # Pad if not evenly divisible
        if len(chunks) > self.num_modalities:
            chunks = list(chunks[:self.num_modalities])
        elif len(chunks) < self.num_modalities:
            chunks = list(chunks) + [chunks[-1]] * (self.num_modalities - len(chunks))

        # Transform each modality
        transformed = []
        for i, (chunk, transform) in enumerate(zip(chunks, self.modality_transforms)):
            transformed.append(transform(chunk))

        # Cross-modal mixing
        mix = torch.softmax(self.mix_matrix, dim=-1)
        stacked = torch.stack(transformed, dim=-2)  # [B, S, M, chunk_dim]
        mixed = torch.einsum('ij,bsjd->bsid', mix, stacked)  # [B, S, M, chunk_dim]

        # Recombine
        blended = mixed.reshape(B, S, -1)[:, :, :D]
        return self.output_proj(blended)


class ImaginationLimb(BaseLimb):
    """
    Generates novel experiences by exploring beyond the training manifold.

    Unlike visualization, imagination is:
    - Generative (doesn't need prior knowledge)
    - Cross-modal (blends sensory representations)
    - Experience-first (not detail-first)
    - "In the moment" — present → possible
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_modalities: int = 4,
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
            limb_name="imagination",
        )

        self.novelty = NoveltyExplorer(hidden_dim)
        self.blender = CrossModalBlender(hidden_dim, num_modalities)

        # Experience synthesizer: VAE-style (mu, logvar → sample)
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)
        self.synthesizer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Novelty score: how far from known territory?
        self.novelty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self._kl_loss = torch.tensor(0.0)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def process(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Push beyond known space
        novel = self.novelty(x)
        # Cross-modal blending
        blended = self.blender(novel)
        # VAE-style experience synthesis
        mu = self.mu_head(blended)
        logvar = self.logvar_head(blended).clamp(-10, 10)
        z = self._reparameterize(mu, logvar)
        # KL divergence loss (encourages diverse imagination)
        self._kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # Synthesize experience
        synthesized = self.synthesizer(z)
        return self.norm(x + synthesized)

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
            confidence = self.novelty_head(pooled).mean().item()

        return output, confidence, None

    def get_kl_loss(self) -> torch.Tensor:
        """KL divergence loss for training (encourages diverse imagination)."""
        return self._kl_loss
