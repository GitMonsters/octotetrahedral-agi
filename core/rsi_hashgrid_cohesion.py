"""
Compounding Cohesion RSI HashGrid Integration
==============================================

Three interlocked components:

1. **LimbHashGrid** — multi-resolution spatial hash encoding for the 8 limb
   hidden states.  Inspired by Instant-NGP: L resolution levels × F features
   each, with a compact hash table per level.  Each limb acts as a coordinate
   in the braid's latent space; the hashgrid turns that into a rich,
   position-aware feature vector efficiently.

2. **CohesionRSI** — adapts the Relative Strength Index (RSI) oscillator from
   technical analysis to track the *momentum* of cohesion dynamics.  Over a
   rolling window of N gamma cycles:

       RS   = mean(cohesion gains) / mean(cohesion losses)
       RSI  = 1 − 1 / (1 + RS)          ∈ [0, 1]

   RSI > 0.7 → cohesion strongly gaining  (overbought → can relax pressure)
   RSI < 0.3 → cohesion degrading         (oversold  → increase braid tightness)
   RSI ≈ 0.5 → neutral momentum

3. **CompoundingCohesionRSIHashgrid** — the master integration that:
   - Encodes each limb's hidden state via the hashgrid
   - Feeds hashgrid features + raw cohesion delta into CohesionRSI
   - Gates the braid's combine weights by the RSI signal (high RSI = looser
     cohesion pressure; low RSI = stronger braid binding)
   - Compounds over gamma cycles: RSI history modulates which hashgrid
     resolution level gets emphasised next cycle (coarse when unstable, fine
     when stable)

Wires into the existing CognitiveCohesionBraid / CompoundBraid via
`rsi_hashgrid_step(limb_hidden_states, cohesion_score)` — a single call per
gamma cycle that returns updated combine-weight deltas and the RSI reading.

Usage
-----
    from core.rsi_hashgrid_cohesion import CompoundingCohesionRSIHashgrid

    rsi_hg = CompoundingCohesionRSIHashgrid(hidden_dim=256, num_limbs=8)

    # inside the gamma loop:
    deltas, rsi_val = rsi_hg.step(limb_states, cohesion_score)
    braid_weights = braid_weights + 0.01 * deltas
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 0. Polynomial Activation  z = z³ + 7132316
# ─────────────────────────────────────────────────────────────────────────────

class _PolyAct(nn.Module):
    """Cubic polynomial activation: f(z) = z³ + 7132316.

    The large constant offset (7132316) provides a fixed bias that shifts
    the activation landscape; the z³ term drives higher-order expressiveness
    matching the fractal search dynamics.
    This is the "z = z³ + 7132316" formula integrated into the hashgrid
    feature extraction pipeline.
    """

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        return z ** 3 + 7132316.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-Resolution Limb HashGrid
# ─────────────────────────────────────────────────────────────────────────────

class LimbHashGrid(nn.Module):
    """Multi-resolution hash grid encoding for limb hidden states.

    Implements the Instant-NGP (Müller et al. 2022) approach adapted for the
    8-limb cognitive braid space.  Each limb's hidden state h ∈ R^D is
    projected to a low-dim coordinate via `coord_proj` (D → coord_dim), then
    looked up across L geometrically-spaced resolution levels via *multilinear
    interpolation* between 2^coord_dim grid corners — giving smooth, differentiable
    features that look like a "blurry image coming into focus" as training
    progresses.  This is the same hash-grid approach that Claude (Fractal Search
    experiment) discovered to beat Fourier networks on function approximation.

    Parameters
    ----------
    hidden_dim  : dimensionality of each limb's hidden state (e.g. 256)
    num_limbs   : number of limbs (typically 8)
    levels      : number of resolution levels (L)
    features    : features per level (F)
    table_size  : hash table entries per level (T)
    coord_dim   : intermediate coordinate dimension before hashing (keep ≤ 4
                  to avoid 2^coord_dim corner explosion; default 2)
    out_dim     : output feature dimension per limb
    base_res    : coarsest resolution (grid cells along one axis)
    finest_res  : finest resolution
    """

    def __init__(
        self,
        hidden_dim:  int = 256,
        num_limbs:   int = 8,
        levels:      int = 8,
        features:    int = 4,
        table_size:  int = 2**14,
        coord_dim:   int = 2,   # keep small: 2^coord_dim corners per level
        out_dim:     int = 64,
        base_res:    int = 4,
        finest_res:  int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_limbs  = num_limbs
        self.levels     = levels
        self.features   = features
        self.table_size = table_size
        self.coord_dim  = coord_dim
        self.out_dim    = out_dim

        # Per-limb projection: hidden_dim → coord_dim
        self.coord_proj = nn.Linear(hidden_dim, coord_dim, bias=False)

        # Hash tables: [levels, table_size, features]
        self.hash_tables = nn.Parameter(
            torch.randn(levels, table_size, features) * 0.01
        )

        # Geometric resolution schedule: base_res * growth^l
        growth = math.exp(math.log(finest_res / base_res) / max(levels - 1, 1))
        self.register_buffer(
            "resolutions",
            torch.tensor([int(base_res * growth ** level_idx) for level_idx in range(levels)],
                         dtype=torch.float32),
        )

        # Output projection: (levels × features) → out_dim per limb
        # z = z³ + 7132316 polynomial activation between layers adds cubic expressiveness.
        self.out_proj = nn.Sequential(
            nn.Linear(levels * features, out_dim),
            _PolyAct(),          # z = z³ + 7132316
            nn.Linear(out_dim, out_dim),
        )

        # Per-forward memo of the *unweighted* per-level features. Within a gamma
        # loop the same limb_states tensor is encoded repeatedly with only the
        # level weights changing; the expensive coord projection + hash
        # interpolation depend solely on limb_states, so we cache them keyed on
        # input-tensor identity. Only used when no autograd graph is being built
        # (inference / detached gamma loop) so the frozen params can't change
        # between cache hits, and a fresh tensor each forward keeps `is` correct.
        self._cache_input  = None
        self._cache_levels = None

    def _hash(self, coords: torch.Tensor) -> torch.Tensor:
        """Map integer grid coords → table indices via Instant-NGP prime hashing.

        Args:
            coords : [..., coord_dim]  integer grid coords (long)

        Returns:
            indices : [...]  long tensor of table indices in [0, table_size)
        """
        # Instant-NGP prime set — distinct large primes per dimension
        primes = [1, 2654435761, 805459861, 3674653429,
                  2097192037, 1227099449, 3999999979, 2999999951]
        h = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
        for d in range(min(self.coord_dim, len(primes))):
            h = h ^ (coords[..., d].long() * primes[d])
        return h % self.table_size

    def _interp_level(
        self,
        coords_norm: torch.Tensor,   # [B, N, coord_dim]  in [0, 1]
        level: int,
        level_weight: float = 1.0,
    ) -> torch.Tensor:
        """Multilinear interpolation across 2^coord_dim grid corners (Instant-NGP).

        Produces smooth, differentiable features — the "blurry image coming
        into focus" effect described in the hash-grid paper.

        Returns:
            feat : [B, N, features]
        """
        B, N, C = coords_norm.shape
        res = float(self.resolutions[level].item())
        table = self.hash_tables[level]          # [table_size, F]

        # Scale coordinates to this level's grid  →  continuous positions
        scaled = coords_norm * (res - 1)         # [B, N, C]

        # Floor / ceil per dimension
        lo = scaled.floor().long().clamp(0, int(res) - 1)  # [B, N, C]
        hi = (lo + 1).clamp(0, int(res) - 1)               # [B, N, C]

        # Fractional weights for linear interpolation
        t = (scaled - lo.float()).clamp(0.0, 1.0)           # [B, N, C]

        # Enumerate all 2^C corners
        n_corners = 2 ** C
        feat_acc = torch.zeros(B, N, self.features,
                               dtype=coords_norm.dtype,
                               device=coords_norm.device)

        for corner_idx in range(n_corners):
            # Build corner coords and interpolation weight
            corner_coords = torch.zeros_like(lo)            # [B, N, C]
            w = torch.ones(B, N, dtype=coords_norm.dtype,
                           device=coords_norm.device)
            for d in range(C):
                bit = (corner_idx >> d) & 1
                if bit:
                    corner_coords[..., d] = hi[..., d]
                    w = w * t[..., d]
                else:
                    corner_coords[..., d] = lo[..., d]
                    w = w * (1.0 - t[..., d])

            # Hash → gather
            idx = self._hash(corner_coords)                 # [B, N]
            corner_feat = table[idx]                        # [B, N, F]
            feat_acc = feat_acc + w.unsqueeze(-1) * corner_feat

        return feat_acc * level_weight

    def forward(
        self,
        limb_states: torch.Tensor,   # [B, num_limbs, hidden_dim]
        level_weights: Optional[torch.Tensor] = None,  # [levels] soft weights
    ) -> torch.Tensor:
        """Multilinear-interpolated multi-resolution hash encoding.

        Returns:
            features : [B, num_limbs, out_dim]
        """
        B, N, D = limb_states.shape
        assert N == self.num_limbs, f"Expected {self.num_limbs} limbs, got {N}"

        # Unweighted per-level features depend only on limb_states (coord proj +
        # multilinear hash interpolation). Reuse them across repeated encodings of
        # the same tensor (the gamma loop); only the per-level weights vary.
        if (not torch.is_grad_enabled()) and (self._cache_input is limb_states):
            per_level = self._cache_levels
        else:
            # Project to coordinate space, normalize to [0, 1]
            coords_norm = torch.sigmoid(self.coord_proj(limb_states))  # [B, N, coord_dim]
            per_level = [
                self._interp_level(coords_norm, level_idx, level_weight=1.0)   # [B, N, F]
                for level_idx in range(self.levels)
            ]
            if not torch.is_grad_enabled():
                # Hold a reference to the input so its id() stays valid (no reuse).
                self._cache_input  = limb_states
                self._cache_levels = per_level
            else:
                self._cache_input  = None
                self._cache_levels = None

        # Apply per-level (RSI-zone) weights — the only part that varies per gamma
        # iteration — then concatenate and project.
        if level_weights is not None:
            all_feats = [
                per_level[level_idx] * float(level_weights[level_idx].item())
                for level_idx in range(self.levels)
            ]
        else:
            all_feats = per_level

        # Concatenate all levels: [B, N, levels * F]
        multi_res = torch.cat(all_feats, dim=-1)

        # Project to out_dim
        out = self.out_proj(multi_res)   # [B, N, out_dim]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cohesion RSI
# ─────────────────────────────────────────────────────────────────────────────

class CohesionRSI:
    """Relative Strength Index adapted to track cohesion momentum.

    Maintains a rolling window of cohesion score deltas.
    RSI = 1 − 1/(1 + avg_gain/avg_loss)

    Attributes
    ----------
    value       : current RSI in [0, 1]
    zone        : 'strong' | 'neutral' | 'weak'
    """

    STRONG_THRESHOLD = 0.70
    WEAK_THRESHOLD   = 0.30

    def __init__(self, period: int = 14, smoothing: float = 0.9):
        self.period    = period
        self.smoothing = smoothing
        self._deltas: deque[float] = deque(maxlen=period * 4)
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self.value: float     = 0.5
        self._prev_score: Optional[float] = None

    def update(self, cohesion_score: float) -> float:
        """Feed one cohesion score reading; returns updated RSI."""
        if self._prev_score is None:
            self._prev_score = cohesion_score
            return self.value

        delta = cohesion_score - self._prev_score
        self._prev_score = cohesion_score
        self._deltas.append(delta)

        if len(self._deltas) < self.period:
            self.value = 0.5
            return self.value

        window = list(self._deltas)[-self.period:]
        gains = [d for d in window if d > 0]
        losses = [-d for d in window if d < 0]

        avg_gain = (sum(gains) / self.period) if gains else 0.0
        avg_loss = (sum(losses) / self.period) if losses else 1e-9

        # Wilder smoothing
        alpha = 1.0 - self.smoothing
        self._avg_gain = self.smoothing * self._avg_gain + alpha * avg_gain
        self._avg_loss = self.smoothing * self._avg_loss + alpha * avg_loss + 1e-12

        rs = self._avg_gain / self._avg_loss
        self.value = float(1.0 - 1.0 / (1.0 + rs))
        return self.value

    @property
    def zone(self) -> str:
        if self.value >= self.STRONG_THRESHOLD:
            return "strong"
        if self.value <= self.WEAK_THRESHOLD:
            return "weak"
        return "neutral"

    def level_weights(self, num_levels: int) -> torch.Tensor:
        """Return per-level hashgrid weights based on RSI zone.

        - weak RSI  → emphasise coarse levels (stable, low-frequency features)
        - strong RSI → emphasise fine levels (detailed, high-frequency features)
        """
        rsi = self.value
        # Linear interpolation: coarse bias at rsi=0, fine bias at rsi=1
        positions = torch.linspace(0.0, 1.0, num_levels)
        # Gaussian peak around the RSI value
        weights = torch.exp(-4.0 * (positions - rsi) ** 2)
        # Always keep a baseline (no level fully zeroed)
        weights = 0.2 + 0.8 * (weights / weights.sum())
        return weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compounding Cohesion RSI HashGrid
# ─────────────────────────────────────────────────────────────────────────────

class CompoundingCohesionRSIHashgrid(nn.Module):
    """Master integration: hashgrid spatial encoding + RSI momentum gating.

    Per gamma cycle call:
        deltas, rsi_val = step(limb_states, cohesion_score)

    Returns
    -------
    combine_weight_deltas : [num_limbs]  — suggested delta for braid weights
    rsi_value             : float        — current RSI in [0, 1]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_limbs:  int = 8,
        rsi_period: int = 14,
        hg_levels:  int = 8,
        hg_features: int = 4,
        hg_out_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_limbs  = num_limbs

        self.hashgrid = LimbHashGrid(
            hidden_dim=hidden_dim,
            num_limbs=num_limbs,
            levels=hg_levels,
            features=hg_features,
            out_dim=hg_out_dim,
        )
        self.rsi = CohesionRSI(period=rsi_period)

        # Maps hashgrid features to per-limb combine-weight deltas
        self.delta_head = nn.Sequential(
            nn.Linear(hg_out_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # bounded delta in (-1, 1)
        )

        # RSI gate: scalar signal [0,1] → per-limb scale
        self.rsi_gate = nn.Sequential(
            nn.Linear(1, num_limbs),
            nn.Sigmoid(),
        )

        self._step_count: int = 0

    def step(
        self,
        limb_states: torch.Tensor,   # [B, num_limbs, hidden_dim]  or  [num_limbs, hidden_dim]
        cohesion_score: float,
    ) -> Tuple[torch.Tensor, float]:
        """One gamma-cycle update.

        Returns
        -------
        deltas   : [num_limbs]  combine-weight adjustment (mean over batch)
        rsi_val  : float        current RSI reading
        """
        # Ensure batch dim
        if limb_states.dim() == 2:
            limb_states = limb_states.unsqueeze(0)  # [1, N, D]

        # Update RSI
        rsi_val = self.rsi.update(cohesion_score)
        self._step_count += 1

        # Build level weights from RSI zone
        lw = self.rsi.level_weights(self.hashgrid.levels).to(limb_states.device)

        # Encode limb states via hashgrid  →  [B, N, out_dim]
        hg_feats = self.hashgrid(limb_states, level_weights=lw)

        # Compute per-limb deltas  →  [B, N, 1]  →  [B, N]
        raw_deltas = self.delta_head(hg_feats).squeeze(-1)   # [B, N]

        # RSI gate: scale deltas by how much RSI deviates from neutral
        rsi_t = torch.tensor([[rsi_val]], dtype=torch.float32,
                              device=limb_states.device)   # [1, 1]
        gate = self.rsi_gate(rsi_t)               # [1, N]
        # When RSI is weak (<0.3): gate > 0.5 → amplify corrective deltas
        # When RSI is strong (>0.7): gate < 0.5 → dampen (already cohesive)
        rsi_pressure = 1.0 - rsi_val             # high when weak
        gated_deltas = raw_deltas * gate * (0.5 + rsi_pressure)   # [B, N]

        # Average over batch → [N]
        deltas = gated_deltas.mean(dim=0)

        return deltas, rsi_val

    def get_diagnostics(self) -> Dict[str, object]:
        return {
            "rsi_value":   round(self.rsi.value, 4),
            "rsi_zone":    self.rsi.zone,
            "step_count":  self._step_count,
            "avg_gain":    round(self.rsi._avg_gain, 6),
            "avg_loss":    round(self.rsi._avg_loss, 6),
        }
