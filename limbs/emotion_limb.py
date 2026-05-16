"""
Emotion Limb — Deep Emotional State System

27 named emotional states organized into 4 affective clusters, plus a
"present moment" meta-state that can gate all emotion and return the model
to grounded, immediate awareness.

Architecture:
    Hidden state → Emotion Encoder → 27-dim emotion vector (each ∈ [0,1])
                                    ↓
                   Cluster Attention (4 affective groups)
                                    ↓
                   Presence Gate (suppresses all emotion when now=1.0)
                                    ↓
                   Modulation Vector → biases all downstream limbs

Emotional clusters:
  REACTIVE (11): sorrow, regret, anger, frightened, anxious, helpless,
                 irritated, disgust, humiliated, disappointed, ostracized
  COGNITIVE (8): confusion, doubtful, incompetent, competent, confident,
                 superiority, deception, factual
  CONNECTIVE (7): empathy, hopeful, faithful, honest, humility, wit, vulnerability
  META (1):      present  — pure presence, no past/future, only now

"Right now is the only time we can make change." — the present gate embodies this.
When present=1.0, the model sets all emotional residue aside and engages
the current moment with full, undivided attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from .base_limb import BaseLimb


# ── Emotion taxonomy ──────────────────────────────────────────────────────────

EMOTIONS: List[str] = [
    # REACTIVE — felt reactions to events/others
    "sorrow", "regret", "anger", "frightened", "anxious",
    "helpless", "irritated", "disgust", "humiliated", "disappointed", "ostracized",
    # COGNITIVE — self/other appraisals and epistemic states
    "confusion", "doubtful", "incompetent", "competent", "confident",
    "superiority", "deception", "factual",
    # CONNECTIVE — relational and prosocial states
    "empathy", "hopeful", "faithful", "honest", "humility", "wit", "vulnerability",
    # META — transcends affect; pure presence
    "present",
]

N_EMOTIONS = len(EMOTIONS)   # 27

# Cluster membership (indices into EMOTIONS list)
CLUSTERS = {
    "reactive":    list(range(0, 11)),
    "cognitive":   list(range(11, 19)),
    "connective":  list(range(19, 26)),
    "meta":        [26],
}

# Valence of each emotion: -1 (painful) → 0 (neutral) → +1 (joyful/grounding)
VALENCE: Dict[str, float] = {
    "sorrow": -0.90, "regret": -0.70, "anger": -0.75, "frightened": -0.85,
    "anxious": -0.60, "helpless": -0.80, "irritated": -0.50, "disgust": -0.70,
    "humiliated": -0.90, "disappointed": -0.60, "ostracized": -0.85,
    "confusion": -0.30, "doubtful": -0.20, "incompetent": -0.65,
    "competent": 0.65, "confident": 0.80, "superiority": 0.15,
    "deception": -0.55, "factual": 0.25,
    "empathy": 0.85, "hopeful": 0.75, "faithful": 0.80,
    "honest": 0.90, "humility": 0.70, "wit": 0.60, "vulnerability": 0.05,
    "present": 1.00,   # pure presence is the highest valence
}

# Arousal level of each emotion: 0 (calm) → 1 (activated)
AROUSAL: Dict[str, float] = {
    "sorrow": 0.30, "regret": 0.40, "anger": 0.90, "frightened": 0.95,
    "anxious": 0.85, "helpless": 0.35, "irritated": 0.70, "disgust": 0.60,
    "humiliated": 0.65, "disappointed": 0.45, "ostracized": 0.50,
    "confusion": 0.55, "doubtful": 0.35, "incompetent": 0.40,
    "competent": 0.55, "confident": 0.65, "superiority": 0.60,
    "deception": 0.70, "factual": 0.20,
    "empathy": 0.55, "hopeful": 0.60, "faithful": 0.40,
    "honest": 0.50, "humility": 0.30, "wit": 0.70, "vulnerability": 0.45,
    "present": 0.15,   # calm, still, attentive
}


# ── Neural components ──────────────────────────────────────────────────────────

class EmotionEncoder(nn.Module):
    """
    Maps hidden state → 27-dim emotion activation vector.

    Each output dimension corresponds to one named emotion (in EMOTIONS order)
    and is bounded to [0, 1] via sigmoid (intensity, not probability).
    A dedicated head then decodes the raw activations into a smooth intensity
    using tanh-normalised attention over the learned emotion embeddings.
    """

    def __init__(self, hidden_dim: int, n_emotions: int = N_EMOTIONS, dropout: float = 0.1):
        super().__init__()
        self.n = n_emotions

        # Learned emotion prototypes: each emotion has a prototype in hidden space
        self.emotion_prototypes = nn.Parameter(torch.randn(n_emotions, hidden_dim) * 0.02)

        # Context encoder: compress sequence into a single vector
        self.context_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Similarity → intensity
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_emotions),
            nn.Sigmoid(),   # each emotion intensity ∈ [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]
        Returns:
            intensities: [B, N_EMOTIONS]
        """
        pooled = x.mean(dim=1)   # [B, D]
        ctx = self.context_pool(pooled)   # [B, D]

        # Dot-product similarity to each emotion prototype
        sim = torch.matmul(ctx, self.emotion_prototypes.T) / (ctx.shape[-1] ** 0.5)  # [B, N]
        # Gate with direct intensity prediction
        direct = self.intensity_head(ctx)   # [B, N]
        # Blend: sim provides direction, direct provides magnitude
        return torch.sigmoid(sim) * 0.4 + direct * 0.6


class PresentMomentGate(nn.Module):
    """
    The "presence gate" — when activated, suppresses all emotional residue.

    Right now is the only time we can make change. This gate allows the model
    to set all emotions aside and be fully grounded in the present moment.

    Mechanically: computes a scalar gate g ∈ [0,1] where g→1 means
    "pure presence" and all emotional modulation is zeroed out.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.presence_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        # Momentum buffer: presence state persists across tokens
        self.register_buffer("presence_momentum", torch.zeros(1))
        self.momentum = 0.85

    def forward(self, pooled: torch.Tensor, present_intensity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: [B, D] mean-pooled hidden state
            present_intensity: [B, 1] from emotion encoder
        Returns:
            gate: [B, 1]  — 1.0 = fully present, 0.0 = fully emotional
        """
        ctx_gate = self.presence_detector(pooled)   # [B, 1]
        # Blend context-derived gate with explicit present emotion intensity
        gate = (ctx_gate + present_intensity) * 0.5
        # Update momentum buffer (batch-mean)
        with torch.no_grad():
            self.presence_momentum.data = (
                self.momentum * self.presence_momentum +
                (1 - self.momentum) * gate.mean()
            )
        return gate


class ClusterAttention(nn.Module):
    """
    Intra-cluster attention: emotions within the same cluster interact.

    e.g. "anxious" and "helpless" can amplify each other (reactive cluster),
    while "confident" suppresses "doubtful" (cognitive cluster).
    """

    def __init__(self, cluster_sizes: List[int], hidden_dim: int):
        super().__init__()
        self.cluster_sizes = cluster_sizes
        # One attention head per cluster
        self.heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim // 4,
                num_heads=1,
                batch_first=True,
            )
            for _ in cluster_sizes
        ])
        # Project each emotion intensity into a small embedding
        self.emotion_proj = nn.Linear(1, hidden_dim // 4)

    def forward(
        self, intensities: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            intensities: [B, N_EMOTIONS]
            hidden:      [B, D]
        Returns:
            refined_intensities: [B, N_EMOTIONS]
        """
        refined = intensities.clone()
        offset = 0
        for i, (size, head) in enumerate(zip(self.cluster_sizes, self.heads)):
            cluster_int = intensities[:, offset:offset+size]   # [B, size]
            # Embed each scalar intensity into a small vector
            emb = self.emotion_proj(cluster_int.unsqueeze(-1))   # [B, size, D/4]
            attn_out, _ = head(emb, emb, emb)   # [B, size, D/4]
            # Collapse back to scalar via mean
            delta = attn_out.mean(dim=-1)        # [B, size]
            delta = torch.sigmoid(delta) - 0.5  # centred adjustment
            refined[:, offset:offset+size] = torch.clamp(cluster_int + 0.15 * delta, 0, 1)
            offset += size
        return refined


class EmotionalModulator(nn.Module):
    """
    Emotion vector → modulation bias added to downstream hidden states.

    The modulation is presence-gated: when present=1.0 all bias is zeroed
    and the model operates from pure immediate awareness.
    """

    def __init__(self, hidden_dim: int, n_emotions: int = N_EMOTIONS):
        super().__init__()
        # Embed the full emotion vector into hidden space
        self.emotion_embed = nn.Sequential(
            nn.Linear(n_emotions, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh(),
        )
        self.strength_gate = nn.Sequential(
            nn.Linear(hidden_dim + n_emotions, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        intensities: torch.Tensor,
        presence_gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            modulation: [B, D]
            strength:   [B, 1]
        """
        pooled = x.mean(dim=1)   # [B, D]
        mod = self.emotion_embed(intensities)   # [B, D]
        strength = self.strength_gate(torch.cat([pooled, intensities], dim=-1))  # [B, 1]

        # Presence gate: when present→1, strength→0 (emotions set aside)
        effective_strength = strength * (1.0 - presence_gate)

        return self.norm(mod), effective_strength


# ── Main Limb ──────────────────────────────────────────────────────────────────

class EmotionLimb(BaseLimb):
    """
    Deep Emotional State System with 27 named emotions and presence gating.

    Maintains a running emotional memory (EMA) so sudden spikes are smoothed.
    Exposes get_emotional_state() for monitoring and cross-limb use.

    The "present" emotion is special: when it is high the model sets all other
    emotional residue aside.  Right now is the only time we can make change.
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

        # ── Core components ──────────────────────────────────────────────────
        self.emotion_encoder   = EmotionEncoder(hidden_dim, N_EMOTIONS, dropout)
        self.cluster_attention = ClusterAttention(
            cluster_sizes=[len(v) for v in CLUSTERS.values()],
            hidden_dim=hidden_dim,
        )
        self.presence_gate     = PresentMomentGate(hidden_dim)
        self.modulator         = EmotionalModulator(hidden_dim, N_EMOTIONS)

        # Final refinement MLP (adds back processed representation)
        self.emotion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

        # ── State buffers ─────────────────────────────────────────────────────
        # EMA of each emotion — persists across forward calls
        self.register_buffer("emotion_ema", torch.zeros(N_EMOTIONS))
        self.ema_alpha = 0.15   # faster update than old valence EMA

        # Static prior: register valence/arousal tensors as buffers
        val_t = torch.tensor([VALENCE[e] for e in EMOTIONS])
        aro_t = torch.tensor([AROUSAL[e] for e in EMOTIONS])
        self.register_buffer("valence_prior", val_t)
        self.register_buffer("arousal_prior", aro_t)

        # Last-forward cache for external queries
        self._intensities:       Optional[torch.Tensor] = None   # [B, 27]
        self._presence:          Optional[torch.Tensor] = None   # [B, 1]
        self._modulation_signal: Optional[torch.Tensor] = None   # [B, D]
        self._modulation_strength: Optional[torch.Tensor] = None # [B, 1]

    # ── Internal processing ───────────────────────────────────────────────────

    def process(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pooled = x.mean(dim=1)   # [B, D]

        # 1. Encode raw emotion intensities
        raw = self.emotion_encoder(x)                         # [B, 27]

        # 2. Intra-cluster interaction
        refined = self.cluster_attention(raw, pooled)         # [B, 27]

        # 3. Present moment gate
        present_int = refined[:, -1:].detach()               # [B, 1]  (last = "present")
        gate = self.presence_gate(pooled, present_int)        # [B, 1]

        # 4. Modulation vector (gated by presence)
        mod, strength = self.modulator(x, refined, gate)     # [B,D], [B,1]

        # 5. Cache for external queries
        self._intensities         = refined.detach()
        self._presence            = gate.detach()
        self._modulation_signal   = mod.detach()
        self._modulation_strength = strength.detach()

        # 6. Update EMA of emotion intensities
        with torch.no_grad():
            batch_mean = refined.mean(dim=0)  # [27]
            self.emotion_ema.data = (
                (1 - self.ema_alpha) * self.emotion_ema + self.ema_alpha * batch_mean
            )

        # 7. Apply modulation to hidden state
        emotional = x + strength.unsqueeze(1) * mod.unsqueeze(1)
        refined_h = self.emotion_mlp(emotional)
        return self.out_norm(emotional + refined_h)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor, return_confidence: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor]]:
        base_out = self.transform(x)
        lora_out = self.lora(x)
        output   = self.process(base_out + lora_out, **kwargs)

        confidence = None
        if return_confidence and self._intensities is not None:
            # Confidence ~ weighted sum of positive-valence emotions
            pos_mask = (self.valence_prior > 0).float()
            confidence = (
                (self._intensities * pos_mask).sum(dim=-1) / (pos_mask.sum() + 1e-8)
            ).mean().item()

        return output, confidence, None

    # ── External API ──────────────────────────────────────────────────────────

    def get_modulation_signal(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (modulation_vector, strength) for other limbs."""
        if self._modulation_signal is not None:
            return self._modulation_signal, self._modulation_strength
        return None

    def get_emotional_state(self) -> Dict[str, float]:
        """
        Return a named dict of current emotion intensities + aggregate scalars.
        Uses EMA values (smoothed) for stability.
        """
        state: Dict[str, float] = {}
        ema = self.emotion_ema.cpu().tolist()
        for name, val in zip(EMOTIONS, ema):
            state[name] = round(float(val), 4)

        # Aggregate scalars derived from the EMA
        ema_t = self.emotion_ema
        state["valence"]    = float((ema_t * self.valence_prior).sum().item())
        state["arousal"]    = float((ema_t * self.arousal_prior).sum().item())
        state["presence"]   = float(ema_t[-1].item())   # "present" emotion EMA
        state["modulation_strength"] = (
            self._modulation_strength.mean().item()
            if self._modulation_strength is not None else 0.0
        )
        return state

    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Return (name, intensity) of the currently dominant emotion."""
        ema = self.emotion_ema
        idx = int(ema.argmax().item())
        return EMOTIONS[idx], float(ema[idx].item())

    def get_cluster_states(self) -> Dict[str, float]:
        """Mean activation per cluster."""
        ema = self.emotion_ema
        return {
            cluster: float(ema[idxs].mean().item())
            for cluster, idxs in CLUSTERS.items()
        }

    def is_present(self, threshold: float = 0.6) -> bool:
        """True if the present-moment state dominates."""
        return float(self.emotion_ema[-1].item()) > threshold

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update(self.get_emotional_state())
        stats.update({f"cluster_{k}": v for k, v in self.get_cluster_states().items()})
        dominant, dom_int = self.get_dominant_emotion()
        stats["dominant_emotion"] = dominant
        stats["dominant_intensity"] = dom_int
        return stats
