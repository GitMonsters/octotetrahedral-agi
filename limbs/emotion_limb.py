"""
Emotion Limb — Valence/Arousal signals that modulate all processing

Emotion is SELF-REFERENTIAL (distinct from empathy which models others).
Produces global modulation signals — motivation, urgency, caution —
that bias every other limb's processing.

Implementation:
- Valence detector: positive ↔ negative affective state
- Arousal detector: calm ↔ excited activation level
- Modulation signal: a small bias vector applied to all limb outputs
- Emotional memory: EMA of recent emotional states for stability
- Motivation signal: drives exploration (curiosity) vs exploitation (certainty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .base_limb import BaseLimb


class AffectDetector(nn.Module):
    """Detects valence (positive/negative) and arousal (calm/excited)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh(),  # [-1, +1]: negative ↔ positive
        )
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # [0, 1]: calm ↔ excited
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = x.mean(dim=1)  # [B, D]
        valence = self.valence_head(pooled)   # [B, 1]
        arousal = self.arousal_head(pooled)    # [B, 1]
        return valence, arousal


class EmotionalModulator(nn.Module):
    """
    Produces a global modulation vector from valence + arousal.
    This vector biases all downstream processing.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Map (valence, arousal) → modulation vector
        self.modulation_net = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Tanh(),  # Bounded modulation
        )
        # Strength gate: how much should emotion influence processing?
        self.strength_gate = nn.Sequential(
            nn.Linear(hidden_dim + 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        valence: torch.Tensor,
        arousal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            modulation: [B, D] bias vector
            strength: [B, 1] how strongly to apply it
        """
        affect = torch.cat([valence, arousal], dim=-1)  # [B, 2]
        modulation = self.modulation_net(affect)  # [B, D]

        pooled = hidden.mean(dim=1)  # [B, D]
        context = torch.cat([pooled, affect], dim=-1)  # [B, D+2]
        strength = self.strength_gate(context)  # [B, 1]

        return modulation, strength


class EmotionLimb(BaseLimb):
    """
    Produces valence/arousal signals and modulation vectors.

    The modulation vector is designed to be added to other limb outputs
    (retrieved via get_modulation_signal() after forward pass).
    This is how emotion influences all other processing.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
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
            limb_name="emotion",
        )

        self.affect_detector = AffectDetector(hidden_dim)
        self.modulator = EmotionalModulator(hidden_dim)

        # Emotional processing layers
        self.emotion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Emotional memory: smooths rapid fluctuations
        self.register_buffer('emotional_ema', torch.zeros(hidden_dim))
        self.ema_momentum = 0.9

        # Stored for other limbs to query
        self._modulation_signal = None
        self._modulation_strength = None
        self._valence = None
        self._arousal = None

    def process(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Detect affect
        valence, arousal = self.affect_detector(x)
        self._valence = valence.detach()
        self._arousal = arousal.detach()

        # Generate modulation signal
        modulation, strength = self.modulator(x, valence, arousal)
        self._modulation_signal = modulation.detach()
        self._modulation_strength = strength.detach()

        # Update emotional memory (EMA)
        with torch.no_grad():
            self.emotional_ema.data = (
                self.ema_momentum * self.emotional_ema +
                (1 - self.ema_momentum) * modulation.mean(dim=0)
            )

        # Emotional processing: hidden states + emotional coloring
        emotional = x + strength.unsqueeze(1) * modulation.unsqueeze(1)
        refined = self.emotion_mlp(emotional)
        return self.norm(emotional + refined)

    def forward(
        self, x: torch.Tensor, return_confidence: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor]]:
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        output = self.process(adapted, **kwargs)

        confidence = None
        if return_confidence and self._arousal is not None:
            # Arousal acts as confidence proxy (higher arousal = more engaged)
            confidence = self._arousal.mean().item()

        return output, confidence, None

    def get_modulation_signal(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get (modulation_vector, strength) for other limbs to use."""
        if self._modulation_signal is not None:
            return self._modulation_signal, self._modulation_strength
        return None

    def get_emotional_state(self) -> Dict[str, float]:
        """Get current emotional state for monitoring."""
        return {
            'valence': self._valence.mean().item() if self._valence is not None else 0.0,
            'arousal': self._arousal.mean().item() if self._arousal is not None else 0.0,
            'modulation_strength': (
                self._modulation_strength.mean().item()
                if self._modulation_strength is not None else 0.0
            ),
            'emotional_memory_norm': self.emotional_ema.norm().item(),
        }

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update(self.get_emotional_state())
        return stats
