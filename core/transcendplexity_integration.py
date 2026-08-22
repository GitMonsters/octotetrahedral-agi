"""TranscendPlexity integration for OctoTetrahedral AGI.

Implements three core concepts from the TranscendPlexity framework:

1. **Alpha-Order Dynamics**: Fractional-order reasoning priority that modulates
   exit conditions, gate strengths, and strategy selection across processing stages.

2. **Compounding Loss Tracking**: Cascading error detection across sequential
   processing steps, used to modulate learning rates and loop depths.

3. **Phase Detection**: Recognition of qualitative regime shifts in reasoning
   dynamics (e.g., pattern-matching → deep symbolic reasoning).

These modules are designed to plug into existing OctoTetrahedral components:
- CompoundLoopController (exit conditions, loop_alpha)
- CompoundBraid (phase rotation, gate strengths)
- MetaCognitionLimb (strategy selection, uncertainty)
- CognitiveGeometryEngine (entropy targets, drift thresholds)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Alpha-Order Dynamics Module
# ---------------------------------------------------------------------------

class AlphaOrderDynamics(nn.Module):
    """Learns fractional-order priorities across reasoning dimensions.

    In TranscendPlexity, each limb has a fractional order αᵢ ∈ (0,1) that
    controls its contribution depth. Here, we learn a soft priority vector
    α ∈ R^n that modulates:
    - Exit gate blending weights (CompoundLoopController)
    - Gate strengths (CompoundBraid)
    - Strategy selection temperature (MetaCognitionLimb)

    The key insight: α is NOT static. It adapts based on the current reasoning
    state, creating a dynamic priority landscape that shifts with input context.
    """

    def __init__(self, hidden_dim: int, num_dimensions: int = 8, temperature: float = 1.0):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.temperature = temperature

        # Adaptive α: conditioned on input representation
        self.alpha_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_dimensions),
        )

        # Learned base priorities (start uniform)
        self.base_alpha = nn.Parameter(torch.ones(num_dimensions) / num_dimensions)

        # History for variance-based importance weighting
        self._alpha_history: deque[torch.Tensor] = deque(maxlen=32)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute alpha-order priorities from current reasoning state.

        Args:
            hidden: [B, L, D] or [B, D] hidden state

        Returns:
            dict with:
                'alpha': [B, num_dimensions] priority weights (sum to 1)
                'alpha_raw': [B, num_dimensions] pre-softmax values
                'priority_entropy': [B] entropy of priority distribution
        """
        # Pool to [B, D] if needed
        if hidden.dim() == 3:
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden

        # Compute adaptive alpha from input context
        alpha_logits = self.alpha_projector(pooled)  # [B, num_dimensions]

        # Mix with base priorities
        alpha_mixed = alpha_logits + self.base_alpha.unsqueeze(0)

        # Apply temperature-scaled softmax
        alpha = F.softmax(alpha_mixed / self.temperature, dim=-1)  # [B, num_dimensions]

        # Track history for variance computation
        self._alpha_history.append(alpha.detach().mean(dim=0))

        # Compute priority entropy (lower = more focused priorities)
        priority_entropy = -(alpha * (alpha + 1e-10).log()).sum(dim=-1)

        return {
            'alpha': alpha,
            'alpha_raw': alpha_mixed,
            'priority_entropy': priority_entropy,
        }

    def get_exit_weights(self, alpha: torch.Tensor) -> torch.Tensor:
        """Convert alpha priorities to exit gate blending weights.

        Higher alpha → more weight on exit (less computation needed).
        Returns weights in [0, 1] for blending exit signals.
        """
        # Use first 2 dimensions as exit blending weights
        # (conceptual mapping: alpha[0] = fast-path weight, alpha[1] = slow-path weight)
        if alpha.shape[-1] >= 2:
            fast_weight = alpha[..., 0]
            slow_weight = alpha[..., 1]
        else:
            fast_weight = alpha[..., 0]
            slow_weight = 1.0 - fast_weight

        return torch.stack([fast_weight, slow_weight], dim=-1)

    def get_braid_strength(self, alpha: torch.Tensor) -> torch.Tensor:
        """Convert alpha priorities to braid strength modulation.

        Higher average alpha → stronger cross-limb coupling.
        """
        return alpha.mean(dim=-1, keepdim=True)  # [B, 1]


# ---------------------------------------------------------------------------
# 2. Compounding Loss Tracker
# ---------------------------------------------------------------------------

class CompoundingLossTracker(nn.Module):
    """Tracks cascading error across sequential processing steps.

    In TranscendPlexity, compounding loss is:
        L_compound(t) = L(t) + α · L(t-1) + α² · L(t-2) + ...

    Here we track:
    - Per-step loss contribution
    - Cumulative compounding loss
    - Loss gradient (is error accelerating or decelerating?)
    - Modulation signals for loop_alpha, learning rate, gate strengths
    """

    def __init__(self, hidden_dim: int, decay: float = 0.9, window: int = 16):
        super().__init__()
        self.decay = decay
        self.window = window

        # Learns to predict loss modulation from hidden state
        self.loss_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Running statistics
        self._loss_history: deque[float] = deque(maxlen=window)
        self._compounding_loss = 0.0
        self._step_count = 0

    def forward(self, hidden: torch.Tensor, step_loss: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """Track compounding loss for the current step.

        Args:
            hidden: [B, L, D] or [B, D] hidden state at current step
            step_loss: [B] optional explicit loss for this step

        Returns:
            dict with:
                'compounding_loss': [B] cumulative compounding loss
                'loss_gradient': [B] rate of change of loss
                'modulation': [B, 1] gate modulation signal (0=clamp, 1=amplify)
        """
        pooled = hidden.mean(dim=1) if hidden.dim() == 3 else hidden

        # Predict loss contribution from representation
        loss_pred = self.loss_predictor(pooled).squeeze(-1)  # [B]

        # Use explicit loss if provided
        if step_loss is not None:
            current_loss = step_loss
        else:
            current_loss = loss_pred

        # Update compounding loss with exponential decay
        self._step_count += 1
        self._compounding_loss = (
            self.decay * self._compounding_loss + current_loss.mean().item()
        )
        self._loss_history.append(self._compounding_loss)

        # Compute loss gradient (is error accelerating?)
        if len(self._loss_history) >= 2:
            recent = list(self._loss_history)
            loss_gradient = recent[-1] - recent[-2]
        else:
            loss_gradient = 0.0

        # Modulation signal:
        # - Low compounding loss → amplify (confident, use more computation)
        # - High compounding loss → clamp (uncertain, be conservative)
        # - Accelerating loss → clamp harder
        modulation = torch.sigmoid(
            -2.0 * current_loss + torch.tensor(loss_gradient, device=pooled.device)
        ).unsqueeze(-1)

        compounding = torch.full(
            (pooled.shape[0],), self._compounding_loss, device=pooled.device
        )
        gradient = torch.full(
            (pooled.shape[0],), loss_gradient, device=pooled.device
        )

        return {
            'compounding_loss': compounding,
            'loss_gradient': gradient,
            'modulation': modulation,
        }

    def get_loop_alpha_scale(self) -> float:
        """Return scale factor for loop_alpha based on compounding loss.

        High compounding loss → reduce loop_alpha (less aggressive updates).
        """
        if self._compounding_loss < 0.1:
            return 1.0  # normal
        elif self._compounding_loss < 0.5:
            return 0.7  # slightly conservative
        else:
            return 0.3  # very conservative

    def reset(self):
        """Reset tracking state for new sequence."""
        self._loss_history.clear()
        self._compounding_loss = 0.0
        self._step_count = 0


# ---------------------------------------------------------------------------
# 3. Phase Detector
# ---------------------------------------------------------------------------

class PhaseDetector(nn.Module):
    """Detects qualitative regime shifts in reasoning dynamics.

    In TranscendPlexity, phase transitions occur when the compounding
    dynamics cross critical thresholds. Here we detect similar regime shifts:

    - EXPLORATION: High entropy, diverse representations, broad attention
    - CONSOLIDATION: Low entropy, focused representations, narrow attention
    - DEEP_REASONING: Stable entropy, structured representations, directed attention
    - OSCILLATION: Erratic entropy, unstable representations, no clear pattern

    Phase detection modulates:
    - Entropy targets (CognitiveGeometryEngine)
    - Drift thresholds (SemanticDriftDetector)
    - Exit conditions (CompoundLoopController)
    """

    PHASES = ['EXPLORATION', 'CONSOLIDATION', 'DEEP_REASONING', 'OSCILLATION']
    NUM_PHASES = len(PHASES)

    def __init__(self, hidden_dim: int, num_heads: int = 4, history_len: int = 16):
        super().__init__()
        self.history_len = history_len

        # Phase classifier: hidden state → phase logits
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.NUM_PHASES),
        )

        # Attention pattern analyzer (for oscillation detection)
        self.attention_analyzer = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # History tracking
        self._entropy_history: deque[float] = deque(maxlen=history_len)
        self._phase_history: deque[int] = deque(maxlen=history_len)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Detect current reasoning phase.

        Args:
            hidden: [B, L, D] hidden state
            attention_weights: [B, H, L, L] optional attention patterns

        Returns:
            dict with:
                'phase_logits': [B, NUM_PHASES] unnormalized phase scores
                'phase_probs': [B, NUM_PHASES] phase probabilities
                'phase_id': [B] detected phase index
                'phase_name': str name of detected phase
                'entropy_signal': [B] entropy of hidden state
                'stability': [B] how stable the phase is (0=erratic, 1=stable)
        """
        B = hidden.shape[0]

        # Pool hidden state
        pooled = hidden.mean(dim=1)  # [B, D]

        # Classify phase
        phase_logits = self.phase_classifier(pooled)  # [B, NUM_PHASES]
        phase_probs = F.softmax(phase_logits, dim=-1)
        phase_id = phase_probs.argmax(dim=-1)  # [B]

        # Compute entropy signal
        entropy_signal = self._compute_entropy(hidden)  # [B]

        # Detect oscillation from attention patterns
        stability = self._compute_stability(hidden, attention_weights)  # [B]

        # If stability is low, override to OSCILLATION phase
        oscillation_mask = stability < 0.3
        phase_id = torch.where(oscillation_mask, torch.full_like(phase_id, 3), phase_id)

        # Update history
        mean_entropy = entropy_signal.mean().item()
        mean_phase = phase_id.mode().values.item() if phase_id.numel() > 0 else 0
        self._entropy_history.append(mean_entropy)
        self._phase_history.append(mean_phase)

        # Determine phase name
        phase_name = self.PHASES[mean_phase]

        return {
            'phase_logits': phase_logits,
            'phase_probs': phase_probs,
            'phase_id': phase_id,
            'phase_name': phase_name,
            'entropy_signal': entropy_signal,
            'stability': stability,
        }

    def _compute_entropy(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute representation entropy (diversity measure)."""
        # Use last-layer representations
        if hidden.dim() == 3:
            # [B, L, D] → softmax over D → entropy over D
            probs = F.softmax(hidden, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean(dim=1)
        else:
            probs = F.softmax(hidden, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        return entropy

    def _compute_stability(
        self,
        hidden: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute phase stability (inverse of oscillation)."""
        B = hidden.shape[0]

        if attention_weights is not None:
            # Analyze attention entropy: low entropy = focused = stable
            attn_entropy = -(attention_weights * (attention_weights + 1e-10).log()).sum(dim=-1)
            # Average over heads and sequence
            stability = 1.0 / (1.0 + attn_entropy.mean(dim=(1, 2)))
        else:
            # Use representation variance as proxy
            if hidden.dim() == 3:
                var = hidden.var(dim=1).mean(dim=-1)  # [B]
            else:
                var = hidden.var(dim=-1)
            # High variance = stable (diverse but structured)
            # Low variance = potentially stuck
            stability = torch.sigmoid(var * 10)

        return stability

    def get_entropy_target(self, phase_id: torch.Tensor) -> torch.Tensor:
        """Return phase-appropriate entropy target.

        EXPLORATION → high target (encourage diversity)
        CONSOLIDATION → low target (encourage focus)
        DEEP_REASONING → medium target (balanced)
        OSCILLATION → medium target (stabilize)
        """
        targets = torch.tensor([3.0, 0.5, 1.5, 1.5], device=phase_id.device)
        return targets[phase_id]

    def get_drift_threshold(self, phase_id: torch.Tensor) -> torch.Tensor:
        """Return phase-appropriate drift threshold.

        EXPLORATION → high threshold (allow large drift)
        CONSOLIDATION → low threshold (tight drift control)
        DEEP_REASONING → medium threshold
        OSCILLATION → very low threshold (stabilize)
        """
        thresholds = torch.tensor([0.8, 0.2, 0.4, 0.1], device=phase_id.device)
        return thresholds[phase_id]


# ---------------------------------------------------------------------------
# 4. Composite TranscendPlexity Controller
# ---------------------------------------------------------------------------

@dataclass
class TranscendPlexityState:
    """Snapshot of TranscendPlexity dynamics at a given step."""
    alpha: Optional[torch.Tensor] = None
    compounding_loss: float = 0.0
    loss_gradient: float = 0.0
    phase_name: str = "EXPLORATION"
    phase_probs: Optional[torch.Tensor] = None
    stability: float = 1.0
    step: int = 0


class TranscendPlexityController(nn.Module):
    """Orchestrates all three TP modules for a processing step.

    Usage in CompoundLoopController:
        tp = TranscendPlexityController(hidden_dim)
        state = tp(hidden, step_loss=loss)
        # Use state.alpha for exit blending
        # Use state.compounding_loss for loop_alpha modulation
        # Use state.phase_name for entropy targets
    """

    def __init__(
        self,
        hidden_dim: int,
        num_dimensions: int = 8,
        alpha_temperature: float = 1.0,
        loss_decay: float = 0.9,
        phase_history_len: int = 16,
    ):
        super().__init__()
        self.alpha_dynamics = AlphaOrderDynamics(
            hidden_dim, num_dimensions, alpha_temperature
        )
        self.compounding_loss = CompoundingLossTracker(
            hidden_dim, decay=loss_decay, window=phase_history_len
        )
        self.phase_detector = PhaseDetector(
            hidden_dim, history_len=phase_history_len
        )

        self._step = 0

    def forward(
        self,
        hidden: torch.Tensor,
        step_loss: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> tuple[TranscendPlexityState, dict[str, torch.Tensor]]:
        """Run all TP modules and return combined state + raw outputs.

        Args:
            hidden: [B, L, D] hidden state
            step_loss: [B] optional explicit loss for this step
            attention_weights: [B, H, L, L] optional attention patterns

        Returns:
            (state, raw_outputs) where:
                state: TranscendPlexityState snapshot
                raw_outputs: dict of all module outputs for backprop
        """
        self._step += 1

        # Alpha-order dynamics
        alpha_out = self.alpha_dynamics(hidden)

        # Compounding loss tracking
        loss_out = self.compounding_loss(hidden, step_loss)

        # Phase detection
        phase_out = self.phase_detector(hidden, attention_weights)

        # Build state
        phase_id = phase_out['phase_id']
        state = TranscendPlexityState(
            alpha=alpha_out['alpha'],
            compounding_loss=loss_out['compounding_loss'].mean().item(),
            loss_gradient=loss_out['loss_gradient'].mean().item(),
            phase_name=phase_out['phase_name'],
            phase_probs=phase_out['phase_probs'],
            stability=phase_out['stability'].mean().item(),
            step=self._step,
        )

        raw_outputs = {
            'alpha': alpha_out,
            'compounding_loss': loss_out,
            'phase': phase_out,
        }

        return state, raw_outputs

    def reset(self):
        """Reset all tracking state for new sequence."""
        self._step = 0
        self.compounding_loss.reset()
        self.phase_detector._entropy_history.clear()
        self.phase_detector._phase_history.clear()
        self.alpha_dynamics._alpha_history.clear()
