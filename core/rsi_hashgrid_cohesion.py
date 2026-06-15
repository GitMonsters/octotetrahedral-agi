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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-Resolution Limb HashGrid
# ─────────────────────────────────────────────────────────────────────────────

class LimbHashGrid(nn.Module):
    """Multi-resolution hash grid encoding for limb hidden states.

    Each limb's hidden state h ∈ R^D is projected down to a low-dim coordinate
    via `coord_proj` (D → coord_dim), then looked up in L hash tables at
    geometrically-spaced resolutions.

    Output per limb: L × F features → projected to out_dim.

    Parameters
    ----------
    hidden_dim  : dimensionality of each limb's hidden state (e.g. 256)
    num_limbs   : number of limbs (typically 8)
    levels      : number of resolution levels (L)
    features    : features per level (F)
    table_size  : hash table entries per level (T)
    coord_dim   : intermediate coordinate dimension before hashing
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
        coord_dim:   int = 4,
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
            torch.tensor([int(base_res * growth ** l) for l in range(levels)],
                         dtype=torch.float32),
        )

        # Output projection: (levels × features) → out_dim per limb
        self.out_proj = nn.Sequential(
            nn.Linear(levels * features, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _hash(self, coords: torch.Tensor, level: int) -> torch.Tensor:
        """Map integer grid coords → table indices via prime hashing.

        Args:
            coords : [B, num_limbs, coord_dim]  integer grid coords
            level  : which level (for table_size lookup)

        Returns:
            indices : [B, num_limbs]  long tensor of table indices
        """
        # FNV-inspired multi-dim hash
        primes = [1, 2654435761, 805459861, 3674653429]
        h = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
        for d in range(min(self.coord_dim, len(primes))):
            h = h ^ (coords[..., d].long() * primes[d])
        return h % self.table_size

    def forward(
        self,
        limb_states: torch.Tensor,   # [B, num_limbs, hidden_dim]
        level_weights: Optional[torch.Tensor] = None,  # [levels] soft weights
    ) -> torch.Tensor:
        """
        Returns:
            features : [B, num_limbs, out_dim]
        """
        B, N, D = limb_states.shape
        assert N == self.num_limbs, f"Expected {self.num_limbs} limbs, got {N}"

        # Project to coordinate space and normalize to [0, 1]
        coords_norm = torch.sigmoid(self.coord_proj(limb_states))  # [B, N, coord_dim]

        all_feats: List[torch.Tensor] = []

        for l in range(self.levels):
            res = self.resolutions[l]  # scalar
            # Map [0,1] coords → integer grid
            grid_coords = (coords_norm * (res - 1)).long()   # [B, N, coord_dim]
            # Clamp to valid range
            grid_coords = grid_coords.clamp(0, int(res.item()) - 1)

            # Hash lookup
            idx = self._hash(grid_coords, l)                 # [B, N]
            # Gather features from hash table
            table = self.hash_tables[l]                      # [table_size, F]
            feat = table[idx]                                # [B, N, F]

            if level_weights is not None:
                feat = feat * level_weights[l]

            all_feats.append(feat)

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

        gains = [d for d in self._deltas if d > 0]
        losses = [-d for d in self._deltas if d < 0]

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
