"""
Manifold Re-Anchoring for the F.A.R.T.S. Neural Braid
=======================================================

Prevents latent space drift by periodically re-anchoring module
representations to a shared "anchor frame" — a set of K stable prototype
vectors that live in braid space z.

Design
------
Every braid module must satisfy the Re-Anchoring Contract:

    r_i(z_i) = (d(P_i(z_i), a_1), ..., d(P_i(z_i), a_K))   # anchor coords

Alignment losses ensure all modules "speak" about the same concepts using
the same coordinates, regardless of their internal parameterization.

Three components
----------------
    AnchorSet         — K learnable prototype vectors in braid space
    ReanchoringLoss   — L_anchor, L_sem_anc, L_dyn_anc  (differentiable)
    ReanchoringController — monitors D_anchor drift; triggers
                            rollback / consolidation / re-anchor as needed

Re-Anchoring Contract (enforced by register_module)
----------------------------------------------------
Every module M_i must provide:
    descriptor.latent_dim         — its own z_i dim
    BraidProjection P_i, P_i^{-1} — already required by MIP
    anchor_encoder R_i            — z_i → R^K  (learned here)

    It will receive from the controller:
        anchor_coords: Tensor[B, K]  — braid anchor coordinates
        drift_status: DriftStatus    — STABLE | DRIFTING | ROLLBACK

Math
----
    d(u, a_k) = 1 - cos(u, a_k)            cosine dissimilarity

    L_anchor-align = E[|| r_new(z_new) - r_core(z_core) ||²]

    L_sem  (re-anchored) = E[|| r_new(ẑ_t+1_new) - r_core(z_ref) ||²]
    L_dyn  (re-anchored) = E[|| r_new(ẑ_t+1_new) - r_core(z_WM)  ||²]

    D_anchor = E_z[ || r_new(z_new) - r_core(z) ||² ]

Controller state machine
------------------------
    PLASTIC   → CANDIDATE  if D_anchor < δ and ΔL_task < 0
    CANDIDATE → CONSOLIDATED if stable over window W
    any       → ROLLBACK    if D_anchor > δ_max
    ROLLBACK  → PLASTIC     after re-anchor step
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Enums & config
# ─────────────────────────────────────────────────────────────────────────────

class DriftStatus(Enum):
    STABLE       = "stable"
    DRIFTING     = "drifting"
    ROLLBACK     = "rollback"
    CONSOLIDATED = "consolidated"


class ConsolidationPhase(Enum):
    PLASTIC      = "plastic"       # exploring; anchor loss enforced loosely
    CANDIDATE    = "candidate"     # performance improved; monitoring stability
    CONSOLIDATED = "consolidated"  # frozen subspace absorbed into braid
    ROLLBACK     = "rollback"      # drift detected; reverting toward snapshot


@dataclass
class ReanchoringConfig:
    # Anchor set
    n_anchors: int  = 32          # K: number of prototype vectors
    braid_dim: int  = 256         # must match MIP braid_dim

    # Drift thresholds
    drift_threshold: float     = 0.15   # δ: candidate consolidation threshold
    drift_max: float           = 0.40   # δ_max: triggers ROLLBACK
    stability_window: int      = 20     # W: steps CANDIDATE must stay stable
    performance_min_delta: float = -0.01  # ΔL_task must be ≤ this to promote

    # Loss weights
    lambda_anchor_align: float = 0.5
    lambda_sem_anc:      float = 0.4
    lambda_dyn_anc:      float = 0.4

    # EMA for anchors & reference model
    anchor_ema_decay:    float = 0.995
    rollback_lr_factor:  float = 0.1    # shrink lr during rollback


# ─────────────────────────────────────────────────────────────────────────────
# Re-Anchoring Contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReanchoringContract:
    """
    Formal contract every braid module must satisfy to participate in
    manifold re-anchoring.

    The controller validates this contract at registration time and
    exposes `anchor_coords` + `drift_status` to the module at each step.
    """
    module_name:   str
    latent_dim:    int               # dim of module's internal z_i space
    braid_dim:     int               # shared braid dim (from MIP)
    domain_tags:   List[str]         # used to bias anchor initialization

    # These are populated by the controller after registration:
    anchor_encoder: Optional[nn.Module] = field(default=None, repr=False)
    snapshot_params: Optional[Dict[str, torch.Tensor]] = field(
        default=None, repr=False
    )
    phase: ConsolidationPhase = ConsolidationPhase.PLASTIC
    stable_steps: int = 0            # consecutive steps below drift_threshold
    last_task_loss: Optional[float] = None

    def is_valid(self) -> bool:
        return (
            self.module_name != ""
            and self.latent_dim > 0
            and self.braid_dim > 0
            and self.anchor_encoder is not None
        )

    def summary(self) -> str:
        return (
            f"ReanchoringContract('{self.module_name}' "
            f"z={self.latent_dim}→braid={self.braid_dim} "
            f"phase={self.phase.value} "
            f"stable_steps={self.stable_steps})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Anchor Set
# ─────────────────────────────────────────────────────────────────────────────

class AnchorSet(nn.Module):
    """
    K learnable prototype vectors in braid space z ∈ R^D.

    Anchors define the shared coordinate frame. All modules express their
    latents in anchor-relative coordinates:

        r(z) = (d(z, a_1), ..., d(z, a_K))

    where d = cosine dissimilarity = 1 - cos(z, a_k).

    EMA update keeps anchors as slow-moving reference points.
    """

    def __init__(self, cfg: ReanchoringConfig):
        super().__init__()
        self.cfg = cfg
        # Learnable anchors (normalized at init)
        anchors = F.normalize(
            torch.randn(cfg.n_anchors, cfg.braid_dim), dim=-1
        )
        self.anchors = nn.Parameter(anchors)

        # EMA buffer (non-differentiable reference copy)
        self.register_buffer("ema_anchors", anchors.clone())

    @torch.no_grad()
    def ema_update(self):
        """Slow-moving EMA copy of anchors — used as stable reference."""
        decay = self.cfg.anchor_ema_decay
        self.ema_anchors.mul_(decay).add_(
            F.normalize(self.anchors.data, dim=-1), alpha=1.0 - decay
        )

    def coords(
        self,
        z: torch.Tensor,       # [B, D]
        use_ema: bool = False,
    ) -> torch.Tensor:         # [B, K]
        """
        Project z into anchor-relative coordinates using cosine dissimilarity.

            r(z)_k = 1 - cos(z, a_k)
        """
        a = self.ema_anchors if use_ema else F.normalize(self.anchors, dim=-1)
        z_n = F.normalize(z, dim=-1)              # [B, D]
        cos_sim = z_n @ a.T                       # [B, K]
        return 1.0 - cos_sim                      # dissimilarity ∈ [0, 2]

    def nearest_anchor(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (nearest anchor vector, index) for each z in batch."""
        coords = self.coords(z)                   # [B, K]
        idx = coords.argmin(dim=-1)               # [B]
        a = F.normalize(self.anchors, dim=-1)
        return a[idx], idx                        # [B, D], [B]


# ─────────────────────────────────────────────────────────────────────────────
# Anchor Encoder  (per-module  z_i → R^K)
# ─────────────────────────────────────────────────────────────────────────────

class AnchorEncoder(nn.Module):
    """
    Module-specific encoder R_i : z_i → anchor coords R^K.

    Projects module latent z_i through BraidProjection P_i (already in MIP),
    then computes anchor-relative coordinates using the shared AnchorSet.

    This is the piece that must be learned per module during Stage 0-1.
    """

    def __init__(
        self,
        module_latent_dim: int,
        braid_dim: int,
        n_anchors: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        # Small MLP: z_i → braid-dim embedding → anchor coords
        self.proj = nn.Sequential(
            nn.Linear(module_latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, braid_dim),
        )
        self.n_anchors = n_anchors

    def forward(
        self,
        z_i: torch.Tensor,             # [B, module_latent_dim]
        anchor_set: AnchorSet,
        use_ema: bool = False,
    ) -> torch.Tensor:                 # [B, K]
        z_braid = self.proj(z_i)       # [B, braid_dim]
        return anchor_set.coords(z_braid, use_ema=use_ema)


# ─────────────────────────────────────────────────────────────────────────────
# Re-Anchoring Losses
# ─────────────────────────────────────────────────────────────────────────────

class ReanchoringLoss(nn.Module):
    """
    Three differentiable alignment losses expressed in anchor space.

    L_anchor_align = E[ || r_new(z_new) - r_core(z_core) ||² ]
    L_sem_anc      = E[ || r_new(ẑ_new_t+1) - r_core(z_ref_t+1) ||² ]
    L_dyn_anc      = E[ || r_new(ẑ_new_t+1) - r_core(z_WM_t+1)  ||² ]

    D_anchor (non-differentiable metric) = mean of L_anchor_align.detach()
    """

    def __init__(self, cfg: ReanchoringConfig):
        super().__init__()
        self.cfg = cfg

    def anchor_align(
        self,
        r_new:  torch.Tensor,   # [B, K] anchor coords from new module
        r_core: torch.Tensor,   # [B, K] anchor coords from core braid
    ) -> Tuple[torch.Tensor, float]:
        """
        L_anchor_align and scalar D_anchor drift metric.
        """
        diff = r_new - r_core
        loss = (diff ** 2).mean()
        d_anchor = loss.detach().item()
        return self.cfg.lambda_anchor_align * loss, d_anchor

    def semantic(
        self,
        r_new_next:  torch.Tensor,  # [B, K] from new module next-state
        r_core_ref:  torch.Tensor,  # [B, K] from reference braid
    ) -> torch.Tensor:
        diff = r_new_next - r_core_ref
        return self.cfg.lambda_sem_anc * (diff ** 2).mean()

    def dynamic(
        self,
        r_new_next: torch.Tensor,   # [B, K] from new module prediction
        r_core_wm:  torch.Tensor,   # [B, K] from world-model prediction
    ) -> torch.Tensor:
        diff = r_new_next - r_core_wm
        return self.cfg.lambda_dyn_anc * (diff ** 2).mean()

    def total(
        self,
        r_new:       torch.Tensor,
        r_core:      torch.Tensor,
        r_new_next:  Optional[torch.Tensor] = None,
        r_core_ref:  Optional[torch.Tensor] = None,
        r_core_wm:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_align, d_anchor = self.anchor_align(r_new, r_core)
        metrics = {
            "l_anchor_align": l_align.item(),
            "d_anchor": d_anchor,
        }
        total = l_align

        if r_new_next is not None and r_core_ref is not None:
            l_sem = self.semantic(r_new_next, r_core_ref)
            total = total + l_sem
            metrics["l_sem_anc"] = l_sem.item()

        if r_new_next is not None and r_core_wm is not None:
            l_dyn = self.dynamic(r_new_next, r_core_wm)
            total = total + l_dyn
            metrics["l_dyn_anc"] = l_dyn.item()

        metrics["l_reanchor_total"] = total.item()
        return total, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Re-Anchoring Controller
# ─────────────────────────────────────────────────────────────────────────────

class ReanchoringController:
    """
    Monitors D_anchor drift for every registered module and drives the
    consolidation state machine:

        PLASTIC → CANDIDATE → CONSOLIDATED
                ↘            ↗
                  ROLLBACK

    Usage
    -----
        ctrl = ReanchoringController(cfg)
        ctrl.register("vision_module", latent_dim=128, braid_dim=256)

        # Each training step:
        status, metrics = ctrl.step(
            name="vision_module",
            z_i=z_vision,           # [B, module_latent_dim]
            z_core=z_braid,         # [B, braid_dim]
            task_loss_delta=delta,  # float; negative = improvement
        )

        # Get total re-anchoring loss to add to training objective:
        loss, breakdown = ctrl.compute_loss("vision_module", z_i, z_core)
    """

    def __init__(self, cfg: Optional[ReanchoringConfig] = None):
        self.cfg      = cfg or ReanchoringConfig()
        self.anchor_set = AnchorSet(self.cfg)
        self.loss_fn    = ReanchoringLoss(self.cfg)

        self._contracts:    Dict[str, ReanchoringContract] = {}
        self._encoders:     Dict[str, AnchorEncoder]       = {}
        self._drift_history: Dict[str, List[float]]        = {}
        self._created_at:    Dict[str, float]              = {}

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        name:        str,
        latent_dim:  int,
        braid_dim:   Optional[int] = None,
        domain_tags: Optional[List[str]] = None,
    ) -> ReanchoringContract:
        """
        Register a module.  Creates its AnchorEncoder and initial snapshot.
        Returns the populated ReanchoringContract.
        """
        bd = braid_dim or self.cfg.braid_dim
        enc = AnchorEncoder(
            module_latent_dim=latent_dim,
            braid_dim=bd,
            n_anchors=self.cfg.n_anchors,
        )
        contract = ReanchoringContract(
            module_name=name,
            latent_dim=latent_dim,
            braid_dim=bd,
            domain_tags=domain_tags or [],
            anchor_encoder=enc,
            phase=ConsolidationPhase.PLASTIC,
        )
        self._contracts[name]     = contract
        self._encoders[name]      = enc
        self._drift_history[name] = []
        self._created_at[name]    = time.time()
        print(f"[Reanchoring] Registered '{name}' "
              f"(z={latent_dim}→braid={bd}, K={self.cfg.n_anchors})")
        return contract

    def get_contract(self, name: str) -> ReanchoringContract:
        if name not in self._contracts:
            raise KeyError(f"Module '{name}' not registered with ReanchoringController")
        return self._contracts[name]

    # ── Anchor coordinate helpers ──────────────────────────────────────────

    def module_coords(
        self,
        name:     str,
        z_i:      torch.Tensor,    # [B, latent_dim]
        use_ema:  bool = False,
    ) -> torch.Tensor:             # [B, K]
        enc = self._encoders[name]
        return enc(z_i, self.anchor_set, use_ema=use_ema)

    def core_coords(
        self,
        z_core:  torch.Tensor,     # [B, braid_dim]
        use_ema: bool = False,
    ) -> torch.Tensor:             # [B, K]
        return self.anchor_set.coords(z_core, use_ema=use_ema)

    # ── Loss computation ───────────────────────────────────────────────────

    def compute_loss(
        self,
        name:        str,
        z_i:         torch.Tensor,                        # [B, latent_dim]
        z_core:      torch.Tensor,                        # [B, braid_dim]
        z_i_next:    Optional[torch.Tensor] = None,       # next-state module latent
        z_core_ref:  Optional[torch.Tensor] = None,       # braid reference next
        z_core_wm:   Optional[torch.Tensor] = None,       # world-model next
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        r_new  = self.module_coords(name, z_i)
        r_core = self.core_coords(z_core)

        r_new_next  = self.module_coords(name, z_i_next) if z_i_next is not None else None
        r_core_ref  = self.core_coords(z_core_ref)       if z_core_ref is not None else None
        r_core_wm   = self.core_coords(z_core_wm)        if z_core_wm is not None else None

        return self.loss_fn.total(r_new, r_core, r_new_next, r_core_ref, r_core_wm)

    # ── Controller step ────────────────────────────────────────────────────

    def step(
        self,
        name:             str,
        z_i:              torch.Tensor,        # [B, latent_dim]
        z_core:           torch.Tensor,        # [B, braid_dim]
        task_loss_delta:  Optional[float] = None,
    ) -> Tuple[DriftStatus, Dict[str, float]]:
        """
        Single monitoring step.  Returns (DriftStatus, metrics_dict).

        Drive this every training step for each registered module.
        """
        contract = self._contracts[name]
        cfg = self.cfg

        # Compute D_anchor (no grad needed for monitoring)
        with torch.no_grad():
            r_new  = self.module_coords(name, z_i,   use_ema=True)
            r_core = self.core_coords(z_core,         use_ema=True)
            _, d_anchor = self.loss_fn.anchor_align(r_new, r_core)

        # Record history
        self._drift_history[name].append(d_anchor)
        if len(self._drift_history[name]) > cfg.stability_window * 2:
            self._drift_history[name] = self._drift_history[name][-cfg.stability_window:]

        # ── State machine ─────────────────────────────────────────────
        prev_phase = contract.phase
        drift_status = DriftStatus.STABLE

        if d_anchor > cfg.drift_max:
            # Immediate rollback regardless of current phase
            contract.phase      = ConsolidationPhase.ROLLBACK
            contract.stable_steps = 0
            drift_status        = DriftStatus.ROLLBACK

        elif contract.phase == ConsolidationPhase.ROLLBACK:
            if d_anchor < cfg.drift_threshold:
                contract.phase = ConsolidationPhase.PLASTIC
                drift_status   = DriftStatus.STABLE
            else:
                drift_status = DriftStatus.ROLLBACK

        elif contract.phase == ConsolidationPhase.PLASTIC:
            if (
                task_loss_delta is not None
                and task_loss_delta <= cfg.performance_min_delta
                and d_anchor < cfg.drift_threshold
            ):
                contract.phase      = ConsolidationPhase.CANDIDATE
                contract.stable_steps = 1
                drift_status        = DriftStatus.STABLE
            elif d_anchor >= cfg.drift_threshold:
                drift_status = DriftStatus.DRIFTING

        elif contract.phase == ConsolidationPhase.CANDIDATE:
            if d_anchor < cfg.drift_threshold:
                contract.stable_steps += 1
                if contract.stable_steps >= cfg.stability_window:
                    contract.phase = ConsolidationPhase.CONSOLIDATED
                    drift_status   = DriftStatus.CONSOLIDATED
                else:
                    drift_status = DriftStatus.STABLE
            else:
                # Drift during candidate phase → back to PLASTIC
                contract.phase       = ConsolidationPhase.PLASTIC
                contract.stable_steps = 0
                drift_status = DriftStatus.DRIFTING

        elif contract.phase == ConsolidationPhase.CONSOLIDATED:
            drift_status = DriftStatus.CONSOLIDATED

        # Save task loss for reference
        if task_loss_delta is not None:
            contract.last_task_loss = task_loss_delta

        # EMA update for anchors
        self.anchor_set.ema_update()

        metrics = {
            "d_anchor":      d_anchor,
            "phase":         contract.phase.value,
            "stable_steps":  contract.stable_steps,
            "drift_status":  drift_status.value,
        }

        if prev_phase != contract.phase:
            print(
                f"[Reanchoring] '{name}': {prev_phase.value} → "
                f"{contract.phase.value}  (D={d_anchor:.4f})"
            )

        return drift_status, metrics

    # ── Rollback ───────────────────────────────────────────────────────────

    def save_snapshot(self, name: str, module: nn.Module) -> None:
        """Save a parameter snapshot for potential rollback."""
        import copy
        self._contracts[name].snapshot_params = {
            k: v.detach().clone() for k, v in module.state_dict().items()
        }

    def rollback(self, name: str, module: nn.Module) -> bool:
        """
        Restore module to last snapshot if available.
        Returns True if rollback was performed.
        """
        contract = self._contracts[name]
        if contract.snapshot_params is None:
            print(f"[Reanchoring] '{name}': no snapshot available for rollback")
            return False
        module.load_state_dict(contract.snapshot_params, strict=False)
        contract.phase        = ConsolidationPhase.PLASTIC
        contract.stable_steps = 0
        print(f"[Reanchoring] '{name}': rolled back to snapshot")
        return True

    # ── Diagnostics ────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Dict]:
        """Return full controller status for all modules."""
        out = {}
        for name, contract in self._contracts.items():
            hist = self._drift_history[name]
            out[name] = {
                "phase":        contract.phase.value,
                "stable_steps": contract.stable_steps,
                "d_anchor_now": hist[-1] if hist else None,
                "d_anchor_mean": sum(hist) / len(hist) if hist else None,
                "has_snapshot": contract.snapshot_params is not None,
                "age_s":        time.time() - self._created_at[name],
            }
        return out

    def recommended_lr_factor(self, name: str) -> float:
        """
        Returns a learning-rate multiplier for the module based on drift phase.
        ROLLBACK → 0.1×, CANDIDATE → 0.5×, PLASTIC → 1.0×, CONSOLIDATED → 0.0×
        """
        phase = self._contracts[name].phase
        return {
            ConsolidationPhase.PLASTIC:      1.0,
            ConsolidationPhase.CANDIDATE:    0.5,
            ConsolidationPhase.CONSOLIDATED: 0.0,
            ConsolidationPhase.ROLLBACK:     self.cfg.rollback_lr_factor,
        }[phase]
