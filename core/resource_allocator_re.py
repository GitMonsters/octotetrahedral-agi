"""
Resource Allocator  (R_ω)
=========================
Implements the Fractional Resource Distribution pillar of the Recursive Engine.

Formal spec
-----------
    c_t = R_ω(z_t, u_t)                        — compute budget at step t
    u_t = uncertainty vector (from WorldModel)

    L_resource = E_{τ_e}[ Σ_t (λ_c · c_t  −  λ_d · g(d_t, c_t)) ]

    where:
        c_t        = compute budget (ponder cost)
        d_t        = task difficulty / epistemic uncertainty
        g(d_t, c_t) = reward for spending more compute when uncertainty is high
                      g = d_t · c_t   (capped by diminishing returns via sqrt)

Optimal policy: spend compute proportionally to epistemic uncertainty and
expected value of information (not uniformly).

Architecture
------------
    ResourceAllocator(z_t, uncertainty, difficulty)
        → c_t  ∈ [0, 1]                       (normalised compute budget)
        → module_gates  ∈ [0, 1]^K            (which modules to activate)
        → plan_depth  : int                    (rollout depth for planning)

    Also wraps / supersedes AdaptiveComputationController by:
    - using latent state z_t (not just raw input)
    - incorporating task-difficulty estimate
    - being trainable via L_resource gradient

Integration
-----------
    trainer.py calls:
        c_t, gates, depth = resource_allocator(z_t, uncertainty, difficulty)
    After collecting trajectories:
        l_res = objective.compute_loss(ponder_cost=ponder_costs, ...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResourceAllocatorConfig:
    latent_dim: int   = 256   # z_t dimensionality (must match WorldModelConfig)
    hidden_dim: int   = 256
    num_modules: int  = 8     # number of cognitive modules (= 8 limbs)

    # Loss weights for L_resource
    lambda_c: float = 0.5     # compute cost weight
    lambda_d: float = 1.0     # uncertainty-value-of-info weight

    # Budget bounds
    min_budget: float = 0.05  # always spend at least 5%
    max_budget: float = 1.0

    # Planning depth bounds
    min_depth: int = 1
    max_depth: int = 8


# ─────────────────────────────────────────────────────────────────────────────
# Resource Allocator
# ─────────────────────────────────────────────────────────────────────────────

class ResourceAllocator(nn.Module):
    """
    R_ω — dynamically allocates compute budget per step.

    Inputs
    ------
    z           : [B, latent_dim]  current latent state
    uncertainty : [B]              epistemic uncertainty from UncertaintyEstimator
    difficulty  : [B]              estimated task difficulty (0-1 scalar)

    Outputs
    -------
    budget          : [B]          normalised compute budget c_t ∈ [min, 1]
    module_gates    : [B, K]       soft gating for K cognitive modules
    plan_depth      : int          rollout depth (max_depth when uncertain)
    resource_loss   : scalar       L_resource for the current step
    """

    def __init__(self, cfg: Optional[ResourceAllocatorConfig] = None):
        super().__init__()
        self.cfg = cfg or ResourceAllocatorConfig()
        c = self.cfg

        # Core allocation network
        self.net = nn.Sequential(
            nn.Linear(c.latent_dim + 2, c.hidden_dim),   # z + uncertainty + difficulty
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
        )

        # Budget head: how much compute to spend
        self.budget_head = nn.Sequential(
            nn.Linear(c.hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Module gating head: which of K modules to activate
        self.gate_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.num_modules),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,           # [B, latent_dim]
        uncertainty: torch.Tensor, # [B]
        difficulty: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        Returns (budget, module_gates, plan_depth, resource_loss).
        """
        cfg = self.cfg

        # Clip inputs to reasonable range
        uncertainty = uncertainty.clamp(0.0, 1.0)
        difficulty  = difficulty.clamp(0.0, 1.0)

        x = torch.cat([
            z,
            uncertainty.unsqueeze(-1),
            difficulty.unsqueeze(-1),
        ], dim=-1)   # [B, latent_dim + 2]

        h = self.net(x)

        # Budget ∈ [min_budget, max_budget]
        raw_budget = self.budget_head(h).squeeze(-1)   # [B]
        budget = cfg.min_budget + raw_budget * (cfg.max_budget - cfg.min_budget)

        # Module gates ∈ [0, 1]
        gates = self.gate_head(h)   # [B, K]

        # Plan depth — integer, derived from budget
        # High budget → more rollout depth (discrete, non-differentiable)
        avg_budget = budget.detach().mean().item()
        plan_depth = max(
            cfg.min_depth,
            min(cfg.max_depth, round(avg_budget * cfg.max_depth))
        )

        # L_resource: penalise raw compute, reward uncertainty-proportional spend
        #   g(d_t, c_t) = sqrt(d_t · c_t)   (diminishing returns)
        combined_difficulty = 0.5 * uncertainty + 0.5 * difficulty   # [B]
        g = torch.sqrt(combined_difficulty * budget + 1e-8)          # [B]
        resource_loss = (cfg.lambda_c * budget - cfg.lambda_d * g).mean()

        return budget, gates, plan_depth, resource_loss

    def ideal_compute(
        self,
        uncertainty: torch.Tensor,  # [B]
        difficulty: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        Analytical ideal budget for a given (uncertainty, difficulty) pair.

        From dL/dc_t = 0:
            λ_c  =  λ_d · ∂g/∂c  =  λ_d / (2 · sqrt(c · d))
            → c*  =  λ_d² · d / (4 · λ_c²)   (clamped to [min, max])
        """
        cfg = self.cfg
        combined = 0.5 * uncertainty + 0.5 * difficulty
        c_star = (cfg.lambda_d ** 2 * combined) / (4.0 * cfg.lambda_c ** 2 + 1e-8)
        return c_star.clamp(cfg.min_budget, cfg.max_budget)

    def compute_batch_loss(
        self,
        z_seq: torch.Tensor,           # [B, T, latent_dim]
        uncertainty_seq: torch.Tensor, # [B, T]
        difficulty_seq: torch.Tensor,  # [B, T]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the formal resource loss over a full trajectory.

            L_resource = E_τ[ Σ_t (λ_c·c_t − λ_d·g(d_t, c_t)) ]

        Returns (loss, breakdown_dict).
        """
        B, T, _ = z_seq.shape
        total_loss = torch.zeros(1, device=z_seq.device)
        total_compute = 0.0
        total_g       = 0.0

        for t in range(T):
            budget, _, _, step_loss = self.forward(
                z_seq[:, t], uncertainty_seq[:, t], difficulty_seq[:, t]
            )
            total_loss   = total_loss + step_loss
            total_compute += budget.mean().item()
            combined = 0.5 * uncertainty_seq[:, t] + 0.5 * difficulty_seq[:, t]
            total_g += torch.sqrt(combined * budget + 1e-8).mean().item()

        loss = total_loss / T
        breakdown = {
            "avg_compute_budget": total_compute / T,
            "avg_g":              total_g / T,
            "resource_loss":      loss.item(),
        }
        return loss, breakdown
