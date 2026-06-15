"""
Fractal Search RSI — Auto-Research Self-Improvement Loop
=========================================================

Inspired by the Fractal Search / AutoResearch paradigm (Karpathy 2025):
  run experiment → evaluate metric → commit result → iterate with mutations.

Applied to the OctoTetrahedral AGI's own hash-grid / cohesion configuration:
the system runs short competitive trials of different `LimbHashGrid` configs,
keeps the best performer, and compounds gains over gamma cycles.

This is *weak RSI*: not a full model rewrite, but meta-optimization of the
hash-grid hyperparameters and coord_proj weights — the AI improving its own
function-approximation machinery.

Three components
----------------
1. **ConfigMutator** — proposes neighbouring configurations (levels, features,
   coord_dim, base/finest_res) via small random mutations + gradient-free
   population search.  Tracks a history of (config, score) pairs.

2. **FractalSearchRSI** — runs mini evaluation trials:
   - Candidate grid encodes N random limb-state batches
   - Score = negative reconstruction loss of a tiny probe network
     (lower loss = better representation quality)
   - Winner replaces the incumbent; loser is discarded.
   - Logs results with scores over time (the "error curve going down").

3. **SelfImprovingCohesionBraid** — thin wrapper that periodically calls
   `FractalSearchRSI.step()` inside the gamma loop, updating the live
   `CompoundingCohesionRSIHashgrid` in-place when a better config is found.

Usage
-----
    from core.fractal_search_rsi import SelfImprovingCohesionBraid

    sib = SelfImprovingCohesionBraid(hidden_dim=256, num_limbs=8)
    sib.attach_to_braid(cohesion_braid)   # replaces cohesion_braid's rsi_hg

    # inside training loop:
    deltas, rsi = sib.gamma_cycle_step(limb_states, cohesion_score)
    # every search_interval steps, SelfImprovingCohesionBraid tries a mutation
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rsi_hashgrid_cohesion import (
    CompoundingCohesionRSIHashgrid,
    LimbHashGrid,
    CohesionRSI,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. HashGrid Configuration + Mutator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HashGridConfig:
    """Mutable hyperparameters for a LimbHashGrid."""
    levels:     int = 8
    features:   int = 4
    coord_dim:  int = 2
    table_size: int = 2**14
    base_res:   int = 4
    finest_res: int = 512
    out_dim:    int = 64

    def n_params(self) -> int:
        return self.levels * self.table_size * self.features

    def to_dict(self) -> Dict:
        return {
            "levels": self.levels,
            "features": self.features,
            "coord_dim": self.coord_dim,
            "table_size": self.table_size,
            "base_res": self.base_res,
            "finest_res": self.finest_res,
            "out_dim": self.out_dim,
        }


class ConfigMutator:
    """Proposes neighbouring HashGridConfig mutations.

    Mutation operators (randomly chosen):
    - ± levels (within [4, 16])
    - ± features (within [2, 8])
    - ± coord_dim (within [1, 4])
    - double/halve finest_res (within [64, 2048])
    - scale table_size (×2 or ÷2)
    """

    LEVELS_RANGE   = (4, 16)
    FEATURES_RANGE = (2, 8)
    COORD_RANGE    = (1, 4)
    FINEST_RANGE   = (64, 2048)

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def mutate(self, cfg: HashGridConfig) -> HashGridConfig:
        """Return a new config with one random mutation applied."""
        new = copy.copy(cfg)
        op = self._rng.choice(["levels", "features", "coord_dim", "finest_res"])
        if op == "levels":
            delta = self._rng.choice([-2, -1, 1, 2])
            new.levels = max(self.LEVELS_RANGE[0],
                             min(self.LEVELS_RANGE[1], new.levels + delta))
        elif op == "features":
            delta = self._rng.choice([-2, -1, 1, 2])
            new.features = max(self.FEATURES_RANGE[0],
                               min(self.FEATURES_RANGE[1], new.features + delta))
        elif op == "coord_dim":
            delta = self._rng.choice([-1, 1])
            new.coord_dim = max(self.COORD_RANGE[0],
                                min(self.COORD_RANGE[1], new.coord_dim + delta))
        elif op == "finest_res":
            factor = self._rng.choice([0.5, 2.0])
            new.finest_res = max(self.FINEST_RANGE[0],
                                 min(self.FINEST_RANGE[1],
                                     int(new.finest_res * factor)))
            new.finest_res = max(new.finest_res, new.base_res * 4)
        return new

    def random_config(self) -> HashGridConfig:
        """Return a completely random config."""
        return HashGridConfig(
            levels=self._rng.choice([4, 6, 8, 12]),
            features=self._rng.choice([2, 4, 8]),
            coord_dim=self._rng.choice([1, 2, 3]),
            finest_res=self._rng.choice([128, 256, 512, 1024]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fractal Search RSI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    config:     HashGridConfig
    score:      float          # higher = better representation quality
    step:       int
    rsi_at_eval: float


class FractalSearchRSI:
    """Meta-optimizer that searches for the best LimbHashGrid config.

    Evaluation metric: representation quality score measured as how well a tiny
    linear probe can reconstruct the original limb states from the hashgrid's
    output.  Higher reconstruction accuracy → better hash-grid config.

    The search follows a simple (1+1)-ES:
      1. Mutate current best config → candidate
      2. Quick probe evaluation on a small batch of limb states
      3. If candidate beats incumbent → replace
      4. Log result

    This mirrors the Fractal Search experiment: "point an AI agent at a metric
    and say make the number go up."  Except here *the system is the agent*.
    """

    def __init__(
        self,
        hidden_dim:       int = 256,
        num_limbs:        int = 8,
        eval_steps:       int = 20,        # gradient steps for probe eval
        probe_lr:         float = 0.01,
        population:       int = 3,         # parallel candidates per search step
        max_history:      int = 200,
    ):
        self.hidden_dim  = hidden_dim
        self.num_limbs   = num_limbs
        self.eval_steps  = eval_steps
        self.probe_lr    = probe_lr
        self.population  = population

        self.mutator     = ConfigMutator()
        self.history:    List[TrialResult] = []
        self.max_history = max_history
        self.step_count  = 0

        # Incumbent
        self.best_config = HashGridConfig()
        self.best_score  = float("-inf")

    # ── Probe evaluation ────────────────────────────────────────────────────

    def _build_grid(self, cfg: HashGridConfig, device: torch.device) -> LimbHashGrid:
        return LimbHashGrid(
            hidden_dim=self.hidden_dim,
            num_limbs=self.num_limbs,
            levels=cfg.levels,
            features=cfg.features,
            table_size=cfg.table_size,
            coord_dim=cfg.coord_dim,
            out_dim=cfg.out_dim,
            base_res=cfg.base_res,
            finest_res=cfg.finest_res,
        ).to(device)

    def _eval_config(
        self,
        cfg: HashGridConfig,
        limb_states: torch.Tensor,   # [B, N, D]
        rsi_val: float,
    ) -> float:
        """Score a config: train a tiny linear probe to reconstruct limb_states.

        Score = -reconstruction_loss (lower loss → higher score).
        Uses a small number of gradient steps so eval is fast.
        """
        device = limb_states.device
        B, N, D = limb_states.shape

        grid = self._build_grid(cfg, device)
        # Linear probe: out_dim → D
        probe = nn.Linear(cfg.out_dim, D, bias=True).to(device)
        opt   = torch.optim.Adam(
            list(grid.parameters()) + list(probe.parameters()),
            lr=self.probe_lr
        )

        grid.train()
        probe.train()
        final_loss = float("inf")
        for _ in range(self.eval_steps):
            opt.zero_grad()
            feats = grid(limb_states.detach())         # [B, N, out_dim]
            recon = probe(feats)                       # [B, N, D]
            loss  = F.mse_loss(recon, limb_states.detach())
            loss.backward()
            opt.step()
            final_loss = loss.item()

        # RSI bonus: if RSI is strong → trust fine resolution; weak → trust coarse
        res_bonus = math.log(cfg.finest_res / 64 + 1) * rsi_val * 0.1
        score = -final_loss + res_bonus
        return score

    # ── Search step ─────────────────────────────────────────────────────────

    def search_step(
        self,
        limb_states: torch.Tensor,   # [B, N, D]
        rsi_val: float,
    ) -> Tuple[HashGridConfig, float, bool]:
        """Run one search iteration.  Returns (best_config, best_score, improved).

        Evaluates `population` mutations + incumbent; keeps winner.
        """
        self.step_count += 1
        device = limb_states.device

        # Always include incumbent as baseline
        candidates = [self.best_config]
        for _ in range(self.population):
            candidates.append(self.mutator.mutate(self.best_config))

        scores = []
        for cfg in candidates:
            try:
                s = self._eval_config(cfg, limb_states, rsi_val)
            except Exception:
                s = float("-inf")
            scores.append(s)

        best_idx   = max(range(len(scores)), key=lambda i: scores[i])
        best_cand  = candidates[best_idx]
        best_score = scores[best_idx]

        improved = (best_score > self.best_score) and (best_idx > 0)
        if improved:
            self.best_config = best_cand
            self.best_score  = best_score

        result = TrialResult(
            config=best_cand,
            score=best_score,
            step=self.step_count,
            rsi_at_eval=rsi_val,
        )
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return best_cand, best_score, improved

    def score_trend(self) -> List[float]:
        """Return score history for monitoring (lower loss = higher score)."""
        return [r.score for r in self.history]

    def get_diagnostics(self) -> Dict:
        trend = self.score_trend()
        return {
            "search_steps": self.step_count,
            "best_score": round(self.best_score, 6),
            "best_config": self.best_config.to_dict(),
            "plateau": len(trend) >= 5 and (max(trend[-5:]) - min(trend[-5:]) < 1e-5),
            "trend_last5": [round(s, 5) for s in trend[-5:]],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Self-Improving Cohesion Braid
# ─────────────────────────────────────────────────────────────────────────────

class SelfImprovingCohesionBraid(nn.Module):
    """Wraps CompoundingCohesionRSIHashgrid with a Fractal Search self-improvement loop.

    Every `search_interval` gamma cycles:
      - Runs FractalSearchRSI.search_step() on the current limb states
      - If a better config is found, rebuilds the live LimbHashGrid in-place
      - Copies over trainable parameters where shapes match (warm start)

    This is the "AI improving its own function-approximation machinery" loop.
    """

    def __init__(
        self,
        hidden_dim:      int = 256,
        num_limbs:       int = 8,
        rsi_period:      int = 14,
        search_interval: int = 50,      # gamma cycles between search steps
        eval_steps:      int = 20,      # probe gradient steps per eval
    ):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.num_limbs       = num_limbs
        self.search_interval = search_interval

        # Live hashgrid integrator
        self.compound = CompoundingCohesionRSIHashgrid(
            hidden_dim=hidden_dim,
            num_limbs=num_limbs,
            rsi_period=rsi_period,
        )

        # Meta-optimizer
        self.searcher = FractalSearchRSI(
            hidden_dim=hidden_dim,
            num_limbs=num_limbs,
            eval_steps=eval_steps,
        )

        self._cycle = 0
        self._last_improved_at = 0
        self._improvement_count = 0

    def attach_to_braid(self, cohesion_braid) -> "SelfImprovingCohesionBraid":
        """Register with a CognitiveCohesionBraid, replacing its rsi_hashgrid."""
        cohesion_braid.attach_rsi_hashgrid(self)
        return self

    # Make SelfImprovingCohesionBraid itself look like a CompoundingCohesionRSIHashgrid
    # so CognitiveCohesionBraid.gamma_cycle_step works transparently.

    def step(
        self,
        limb_states: torch.Tensor,
        cohesion_score: float,
    ) -> Tuple[torch.Tensor, float]:
        """One gamma-cycle step: encode → RSI → (optionally) search."""
        self._cycle += 1

        # Standard forward pass
        deltas, rsi_val = self.compound.step(limb_states, cohesion_score)

        # Periodic self-improvement search
        if self._cycle % self.search_interval == 0:
            self._run_search(limb_states, rsi_val)

        return deltas, rsi_val

    def get_diagnostics(self) -> Dict:
        diag = self.compound.get_diagnostics()
        diag["fractal_search"] = self.searcher.get_diagnostics()
        diag["improvements"] = self._improvement_count
        diag["cycles"] = self._cycle
        return diag

    def _run_search(self, limb_states: torch.Tensor, rsi_val: float) -> None:
        """Run one Fractal Search step and hot-swap hashgrid if better config found."""
        # Use a small detached batch for evaluation (no graph leakage)
        states_eval = limb_states.detach()
        if states_eval.shape[0] > 2:
            states_eval = states_eval[:2]

        best_cfg, best_score, improved = self.searcher.search_step(states_eval, rsi_val)

        if improved:
            self._improvement_count += 1
            self._last_improved_at  = self._cycle
            self._hot_swap_hashgrid(best_cfg, limb_states.device)

    def _hot_swap_hashgrid(self, cfg: HashGridConfig, device: torch.device) -> None:
        """Replace the live LimbHashGrid with a better-configured one.

        Copies hash_tables where shape matches (warm start); reinitialises
        coord_proj and out_proj (they'll fine-tune quickly).
        """
        old_grid = self.compound.hashgrid
        new_grid = LimbHashGrid(
            hidden_dim=self.hidden_dim,
            num_limbs=self.num_limbs,
            levels=cfg.levels,
            features=cfg.features,
            table_size=cfg.table_size,
            coord_dim=cfg.coord_dim,
            out_dim=cfg.out_dim,
            base_res=cfg.base_res,
            finest_res=cfg.finest_res,
        ).to(device)

        # Warm-start: copy hash table entries where shapes match
        old_t = old_grid.hash_tables.data
        new_t = new_grid.hash_tables.data
        min_levels = min(old_t.shape[0], new_t.shape[0])
        min_table  = min(old_t.shape[1], new_t.shape[1])
        min_feat   = min(old_t.shape[2], new_t.shape[2])
        with torch.no_grad():
            new_t[:min_levels, :min_table, :min_feat] = \
                old_t[:min_levels, :min_table, :min_feat]
        new_grid.hash_tables.data = new_t

        self.compound.hashgrid = new_grid
