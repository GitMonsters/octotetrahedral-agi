"""
Compound Cohesion Integration — Recursive Agentic Braid Layer
==============================================================

This is the master integration point that makes the OctoTetrahedral model
*fully* a compounding cohesion recursive agentic system.

Architecture
------------
                        ┌──────────────────────────────────┐
  8 limb states ──────► │  CompoundCohesionIntegrator      │
                        │                                  │
                        │  1. Per-limb RSI gating          │
                        │     each limb's contribution     │
                        │     scaled by its own momentum   │
                        │                                  │
                        │  2. N recursive gamma iterations │
                        │     SelfImprovingCohesionBraid   │
                        │     ├─ LimbHashGrid (Instant-NGP)│
                        │     ├─ CohesionRSI               │
                        │     └─ FractalSearchRSI (meta)   │
                        │                                  │
                        │  3. Compounding offset buffer    │
                        │     EWMA across forward passes   │
                        │     δ_t = 0.95*δ_{t-1} + 0.05*Δ │
                        │                                  │
                        │  4. Agentic feedback signal      │
                        │     RSI value → braid routing    │
                        └──────────────────────────────────┘
                              │           │         │
                         gated limbs   rsi_val   gate_vec

Every component is *compounding*: RSI offsets accumulate across steps,
per-limb trackers compound their own history, and FractalSearchRSI
recursively improves the hashgrid configuration over the model's lifetime.

Usage (in model.py)
-------------------
    from core.compound_cohesion_integration import CompoundCohesionIntegrator

    # __init__:
    self.cohesion_integrator = CompoundCohesionIntegrator(
        hidden_dim=self.hidden_dim, num_limbs=8
    )

    # forward (after limbs run, before / alongside compound braid):
    gated, rsi_val, gate_vec = self.cohesion_integrator(
        limb_states=[memory_out, spatial_out, language_out, meta_out,
                     reasoning_out, perception_echo, dream_out, empathy_out],
        cohesion_score=_braid_conf,
    )
    # use gated[i] instead of original limb_out[i] for compound braid
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .fractal_search_rsi import SelfImprovingCohesionBraid
from .rsi_hashgrid_cohesion import CohesionRSI


# Canonical names for the 8 cognitive-core limbs
CORE_LIMB_NAMES = [
    "memory", "spatial", "language", "meta",
    "reasoning", "perception", "dream", "empathy",
]


class CompoundCohesionIntegrator(nn.Module):
    """Recursive agentic integration of all cohesion/RSI subsystems.

    Replaces the standalone ``CompoundingCohesionRSIHashgrid`` in model.py
    with a fully integrated loop:

    * **Per-limb RSI trackers** — each of the 8 cognitive-core limbs has its
      own ``CohesionRSI`` instance, tracking the momentum of that limb's
      activation norm.  The RSI value for each limb becomes its gate factor.

    * **N recursive gamma iterations** — rather than a single hashgrid step,
      the integrator runs ``gamma_iters`` cycles of
      ``SelfImprovingCohesionBraid.step()``, accumulating delta updates.
      Each iteration conditions on the *previous* RSI reading, creating a
      closed feedback loop within a single forward pass.

    * **Compounding offset buffer** — ``_braid_offsets`` is an EMA-smoothed
      accumulation of RSI deltas across forward passes.  Old influences decay
      (α = 0.95) while new deltas compound (β = 0.05).  This makes the
      model's gating *history-aware* — not just reactive to the current step.

    * **Agentic feedback** — the final RSI value and gate vector are returned
      for downstream use: gating the CompoundBraid signal, MoE routing, and
      any other attention mechanisms that benefit from cohesion pressure.

    Parameters
    ----------
    hidden_dim      : limb hidden state dimension
    num_limbs       : number of cognitive-core limbs (default 8)
    gamma_iters     : recursive iteration count per forward pass (default 3)
    search_interval : how often FractalSearchRSI runs meta-search
    rsi_period      : rolling window for CohesionRSI oscillator
    gate_scale      : amplitude of RSI gating (output in [1-s, 1+s])
    adaptive_gamma  : stop the gamma loop early once it converges / stalls
    gamma_conv_eps  : convergence threshold for adaptive early-stop
    gamma_min_iters : minimum gamma iterations before early-stop may fire
    """

    def __init__(
        self,
        hidden_dim:      int = 256,
        num_limbs:       int = 8,
        gamma_iters:     int = 3,
        search_interval: int = 200,
        rsi_period:      int = 14,
        gate_scale:      float = 0.3,
        adaptive_gamma:  bool = True,
        gamma_conv_eps:  float = 1e-3,
        gamma_min_iters: int = 1,
    ):
        super().__init__()
        assert num_limbs == len(CORE_LIMB_NAMES), (
            f"num_limbs must be {len(CORE_LIMB_NAMES)}"
        )
        self.hidden_dim      = hidden_dim
        self.num_limbs       = num_limbs
        self.gamma_iters     = gamma_iters
        self.gate_scale      = gate_scale
        self.adaptive_gamma  = adaptive_gamma
        self.gamma_conv_eps  = gamma_conv_eps
        self.gamma_min_iters = max(1, int(gamma_min_iters))

        # SelfImproving braid: RSI + hashgrid + FractalSearch meta-optimizer
        self.sib = SelfImprovingCohesionBraid(
            hidden_dim=hidden_dim,
            num_limbs=num_limbs,
            rsi_period=rsi_period,
            search_interval=search_interval,
            eval_steps=5,
        )

        # Per-limb RSI trackers (pure Python, no grad)
        self._limb_rsi: Dict[str, CohesionRSI] = {
            name: CohesionRSI(period=rsi_period)
            for name in CORE_LIMB_NAMES
        }

        # Compounding offset buffer: EMA-smoothed RSI deltas [num_limbs]
        self.register_buffer("_braid_offsets", torch.zeros(num_limbs))

        self._forward_count = 0
        self._last_iters_run = gamma_iters

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        limb_states: List[torch.Tensor],      # list of [B, seq, D] per limb
        cohesion_score: float = 0.5,
    ) -> Tuple[List[torch.Tensor], float, torch.Tensor]:
        """Recursive agentic integration step.

        Parameters
        ----------
        limb_states    : list of 8 tensors [B, seq, D], one per core limb
        cohesion_score : scalar cohesion from braid (0..1); updated each iter

        Returns
        -------
        gated_states   : list of 8 tensors [B, seq, D] — RSI-gated limb outs
        rsi_val        : final RSI reading after gamma_iters (0..1)
        gate_vec       : [num_limbs] gate factors used (for braid signal)
        """
        assert len(limb_states) == self.num_limbs
        self._forward_count += 1
        device = limb_states[0].device

        # ── Step 1: update per-limb RSI trackers ──────────────────────────
        for i, name in enumerate(CORE_LIMB_NAMES):
            limb_conf = float(
                limb_states[i].detach().norm(dim=-1).mean().item()
            )
            self._limb_rsi[name].update(limb_conf)

        # Per-limb RSI values → [num_limbs]
        limb_rsi_vals = torch.tensor(
            [self._limb_rsi[n].value for n in CORE_LIMB_NAMES],
            dtype=torch.float32, device=device,
        )

        # ── Step 2: N recursive gamma iterations ──────────────────────────
        # Build mean-pooled limb state tensor [B, num_limbs, D]
        _core = torch.stack(
            [ls.mean(dim=1) for ls in limb_states], dim=1
        )  # [B, 8, D]

        rsi_val = cohesion_score
        prev_rsi = cohesion_score
        accumulated_deltas = torch.zeros(self.num_limbs, device=device)
        _prev_deltas = None
        _iters_run = 0

        # Recursive feedback signal. Feeding the bounded RSI output (centred on
        # 0.5) straight back as the next cohesion score collapses the deltas to
        # ~0 after the first iteration, pinning the oscillator at neutral. We
        # instead blend the *external* cohesion drive with the RSI reading so the
        # loop stays closed (conditions on prior RSI) without losing momentum.
        _feedback = cohesion_score

        for _iter in range(self.gamma_iters):
            try:
                _deltas, rsi_val = self.sib.step(_core, cohesion_score=_feedback)
            except Exception:
                break
            accumulated_deltas = accumulated_deltas + _deltas.detach()
            _iters_run += 1
            _feedback = 0.5 * cohesion_score + 0.5 * rsi_val

            # Adaptive compute + cognitive-loop-trap safeguard. The gamma loop is
            # the dominant braid cost and scales linearly with gamma_iters. Once
            # the recursion stabilises — consecutive deltas stop changing AND the
            # RSI reading stalls — further iterations only re-add the same signal,
            # so we stop early. Behaviour is preserved in the active regime (the
            # loop runs the full gamma_iters whenever it is still doing work).
            if (self.adaptive_gamma and _prev_deltas is not None
                    and _iters_run >= self.gamma_min_iters):
                _delta_settle = float(
                    (_deltas.detach() - _prev_deltas).abs().mean().item()
                )
                _rsi_settle = abs(float(rsi_val) - float(prev_rsi))
                if (_delta_settle < self.gamma_conv_eps
                        and _rsi_settle < self.gamma_conv_eps):
                    break
            _prev_deltas = _deltas.detach()
            prev_rsi = rsi_val

        # Mean delta over the iterations actually executed. Adaptive early-stop
        # leaves this ≈ unchanged vs the full loop because the skipped iterations
        # contribute near-identical deltas; the active regime runs all gamma_iters
        # so _iters_run == gamma_iters and the result is bit-for-bit the same.
        accumulated_deltas = accumulated_deltas / max(_iters_run, 1)
        self._last_iters_run = _iters_run

        # ── Step 3: Compound offset buffer (EMA across forward passes) ────
        with torch.no_grad():
            self._braid_offsets = (
                0.95 * self._braid_offsets
                + 0.05 * accumulated_deltas
            )

        # ── Step 4: Gate vector — blend limb_rsi + compounding offsets ────
        # gate = sigmoid(offset) ∈ (0, 1) → scale to [1-s, 1+s]
        raw_gate = torch.sigmoid(self._braid_offsets)          # [8] ∈ (0,1)
        # Blend: high RSI → trust compounding gate; low RSI → stay near 1.0
        blended_gate = (
            limb_rsi_vals * raw_gate
            + (1.0 - limb_rsi_vals) * torch.ones_like(raw_gate) * 0.5
        )
        gate_vec = 1.0 - self.gate_scale + 2.0 * self.gate_scale * blended_gate
        # gate_vec[i] ∈ [1-scale, 1+scale] ≈ [0.7, 1.3]

        # ── Step 5: Apply gates to limb state tensors ─────────────────────
        gated_states = [
            limb_states[i] * gate_vec[i]
            for i in range(self.num_limbs)
        ]

        return gated_states, rsi_val, gate_vec

    def get_diagnostics(self) -> Dict:
        """Full diagnostic dict for logging / cohesion_score() output."""
        return {
            "forward_count":  self._forward_count,
            "gamma_iters":    self.gamma_iters,
            "iters_run":      self._last_iters_run,
            "adaptive_gamma": self.adaptive_gamma,
            "rsi_val":        round(self.sib.compound.rsi.value, 4),
            "rsi_zone":       self.sib.compound.rsi.zone,
            "improvements":   self.sib._improvement_count,
            "braid_offsets":  [round(float(x), 4) for x in self._braid_offsets],
            "per_limb_rsi": {
                name: round(self._limb_rsi[name].value, 4)
                for name in CORE_LIMB_NAMES
            },
            "fractal_search": self.sib.searcher.get_diagnostics(),
        }

    def rsi_braid_signal(self, braid_signal: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Enrich a braid_signal tensor with RSI cohesion pressure.

        Scales the braid signal by (1 + 0.1 * rsi_val) — when cohesion is
        strong, braid signal amplified; when weak, attenuated.
        """
        if braid_signal is None:
            return None
        rsi = self.sib.compound.rsi.value
        return braid_signal * (1.0 + 0.1 * (rsi - 0.5))
