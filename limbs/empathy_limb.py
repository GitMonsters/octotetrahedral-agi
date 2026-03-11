"""
Empathy Limb — Theory of Mind / Agent Modeling

Simulates another agent's internal state given observed behavior.
Not the same as emotion (which is self-referential) — empathy is
modeling what ANOTHER entity feels, thinks, or intends.

Implementation:
- Agent state encoder: builds a model of the "other"
- Perspective projection: transforms own hidden state into estimated other's state
- Prediction head: what would the other agent do/say next?
- Divergence tracking: how different is my model from theirs?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .base_limb import BaseLimb


class AgentStateEncoder(nn.Module):
    """Encodes observed behavior into an estimated agent state."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Attend to observed behavior to extract agent model
        self.behavior_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.agent_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observed: [B, S, D] — observed context (what the other agent said/did)
        Returns:
            agent_state: [B, D] — estimated internal state of the other
        """
        attn_out, _ = self.behavior_attn(observed, observed, observed)
        pooled = attn_out.mean(dim=1)  # [B, D]
        return self.norm(self.agent_projector(pooled))


class PerspectiveProjector(nn.Module):
    """
    Projects own hidden state into the estimated frame of another agent.
    "If I were them, how would I see this?"
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Rotation-like transformation (perspective shift)
        self.perspective_transform = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, own_state: torch.Tensor, agent_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            own_state: [B, S, D] — my representation
            agent_state: [B, D] — their estimated state
        Returns:
            shifted: [B, S, D] — my representation from their perspective
        """
        agent_expanded = agent_state.unsqueeze(1).expand_as(own_state)
        combined = torch.cat([own_state, agent_expanded], dim=-1)
        perspective = self.perspective_transform(combined)
        gate = self.gate(combined)
        # Gated blend: partially shift perspective
        return own_state * (1 - gate) + perspective * gate


class EmpathyLimb(BaseLimb):
    """
    Models another agent's internal state (Theory of Mind).

    Distinct from Emotion (which is self-referential).
    Empathy asks: "What are THEY feeling/thinking/intending?"
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
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
            limb_name="empathy",
        )

        self.agent_encoder = AgentStateEncoder(hidden_dim, num_heads, dropout)
        self.perspective = PerspectiveProjector(hidden_dim)

        # Prediction: what would the other agent output?
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Empathy strength: how well can we model the other?
        self.empathy_score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def process(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Encode observed behavior into agent model
        agent_state = self.agent_encoder(x)  # [B, D]
        # Shift perspective
        shifted = self.perspective(x, agent_state)
        # Predict what the other agent would produce
        predicted = self.prediction_head(shifted)
        # Blend: enrich own representation with empathic understanding
        return self.norm(x + 0.3 * predicted)

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
            confidence = self.empathy_score_head(pooled).mean().item()

        return output, confidence, None
