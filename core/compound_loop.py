"""
Compound Loop Controller

Adaptive looped reasoning inspired by Ouro (Scaling Latent Reasoning via
Looped Language Models). Wraps the limb processing pipeline in a learnable
loop with:
  - Per-step exit gate (sigmoid + survival probability math)
  - Entropy regularization to prevent collapse to single loop
  - Weighted loss across all loop depths during training
  - Adaptive early exit during inference

"Compound" because it loops the entire compound braid + limb pipeline,
giving the model variable compute per input depending on difficulty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CompoundLoopConfig:
    max_loops: int = 4
    exit_threshold: float = 0.5
    entropy_beta: float = 0.1  # KL regularization strength
    prior: str = 'uniform'     # 'uniform' >> 'geometric' (paper finding)
    warmup_loops: int = 0      # force minimum loops before exit gate activates


class ExitGate(nn.Module):
    """Learned exit gate: Linear → Sigmoid → P(exit at this step)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-batch exit probability from pooled representation."""
        # x: [batch, seq, hidden] → pool → [batch, 1]
        pooled = x.mean(dim=1)
        return self.gate(pooled).squeeze(-1)  # [batch]


class CompoundLoopController(nn.Module):
    """
    Manages adaptive looping over a processing block.

    During training: runs all loops, computes weighted loss using exit
    distribution, adds entropy regularization.

    During inference: exits early when CDF > threshold.
    """

    def __init__(self, hidden_dim: int, config: Optional[CompoundLoopConfig] = None):
        super().__init__()
        self.config = config or CompoundLoopConfig()
        self.max_loops = self.config.max_loops
        self.exit_threshold = self.config.exit_threshold
        self.entropy_beta = self.config.entropy_beta
        self.warmup_loops = self.config.warmup_loops

        # Exit gates for loops 0..max_loops-2 (final step force-exits with remaining mass)
        self.exit_gates = nn.ModuleList([
            ExitGate(hidden_dim) for _ in range(max(1, self.max_loops - 1))
        ])

        # Lightweight loop-step embedding so the block knows which iteration it's on
        self.loop_embed = nn.Embedding(self.max_loops, hidden_dim)

        # Layer norm to stabilize looped representations
        self.loop_norm = nn.LayerNorm(hidden_dim)

        # Residual scaling — start small so early training is stable
        self.loop_alpha = nn.Parameter(torch.tensor(0.1))

        # Track stats
        self._last_loop_count = 0
        self._last_exit_dist: List[float] = []

    def forward(
        self,
        x: torch.Tensor,
        process_fn,
        process_kwargs: Optional[Dict] = None
    ) -> Dict:
        """
        Run process_fn in an adaptive loop.

        Args:
            x: Input tensor [batch, seq, hidden]
            process_fn: Callable(x, loop_idx, **kwargs) → x_processed
            process_kwargs: Extra kwargs passed to process_fn

        Returns:
            Dict with 'output', 'loop_count', 'exit_distribution',
            'entropy_loss', 'loop_outputs' (training only)
        """
        process_kwargs = process_kwargs or {}
        B = x.size(0)
        device = x.device

        exit_probs = []      # sigmoid outputs per step
        loop_outputs = []    # processed representations per step
        survival = torch.ones(B, device=device)
        cdf = torch.zeros(B, device=device)

        base_x = x  # preserve for residual

        for loop_idx in range(self.max_loops):
            # Add loop position embedding
            loop_emb = self.loop_embed(
                torch.tensor(loop_idx, device=device)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
            x_input = x + loop_emb

            # Process this loop iteration
            x_processed = process_fn(x_input, loop_idx, **process_kwargs)

            # Residual connection with learnable scaling
            x = self.loop_norm(base_x + self.loop_alpha * (x_processed - base_x))

            loop_outputs.append(x)

            # Exit gate (last step has no gate — it gets remaining mass)
            is_last = (loop_idx == self.max_loops - 1)
            if not is_last:
                p_exit = self.exit_gates[loop_idx](x)  # [batch]
                p_exit = p_exit.clamp(1e-6, 1 - 1e-6)
                exit_probs.append(p_exit)

                unconditional = survival * p_exit
                cdf = cdf + unconditional
                survival = survival * (1 - p_exit)

                # Inference early exit (after warmup loops)
                if not self.training and loop_idx >= self.warmup_loops:
                    if cdf.mean() > self.exit_threshold:
                        self._last_loop_count = loop_idx + 1
                        return {
                            'output': x,
                            'loop_count': loop_idx + 1,
                            'exit_distribution': [p.mean().item() for p in exit_probs],
                            'entropy_loss': torch.tensor(0.0, device=device),
                            'loop_outputs': None
                        }

        # --- Training: compute proper exit distribution & weighted output ---
        # P(t) = S(t) * p_exit(t) for gated steps, remaining mass for final step
        # exit_probs has max_loops-1 entries (one per gate)
        P = []
        s = torch.ones(B, device=device)
        for p in exit_probs:  # all gated steps
            P.append(s * p)
            s = s * (1 - p)
        P.append(s)  # remaining mass → final step (forced exit, no gate)
        # len(P) == max_loops == len(loop_outputs)

        # Weighted combination of all loop outputs
        # P[i]: [batch], loop_outputs[i]: [batch, seq, hidden]
        weighted_output = torch.zeros_like(loop_outputs[0])
        for p_t, out_t in zip(P, loop_outputs):
            weighted_output = weighted_output + p_t.unsqueeze(-1).unsqueeze(-1) * out_t

        # Entropy regularization (KL vs uniform prior)
        entropy_loss = self._entropy_regularization(P, device)

        # Track stats
        self._last_loop_count = self.max_loops
        self._last_exit_dist = [p.mean().item() for p in P]

        return {
            'output': weighted_output,
            'loop_count': self.max_loops,
            'exit_distribution': self._last_exit_dist,
            'entropy_loss': entropy_loss,
            'loop_outputs': loop_outputs
        }

    def _entropy_regularization(
        self, P: List[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """KL(exit_dist || uniform_prior) — penalizes collapsed distributions."""
        K = len(P)
        if K <= 1:
            return torch.tensor(0.0, device=device)

        # Stack: [batch, K]
        exit_dist = torch.stack(P, dim=-1)
        exit_dist = exit_dist.clamp(min=1e-8)  # prevent log(0)

        # Uniform prior
        uniform = torch.ones(K, device=device) / K

        # KL divergence: sum p * log(p/q)
        kl = (exit_dist * (exit_dist.log() - uniform.log())).sum(dim=-1)

        return self.entropy_beta * kl.mean()

    def get_stats(self) -> Dict:
        return {
            'last_loop_count': self._last_loop_count,
            'last_exit_distribution': self._last_exit_dist,
            'loop_alpha': self.loop_alpha.item()
        }
