"""
Module Integration Protocol  (MIP)
====================================
Formal implementation of the braid Module Integration Protocol.

Spec
----
New cognitive module M_new passes through four phases:
    Stage 0  — Registration     (type, I/O signatures, projection learning)
    Stage 1  — Shadow mode      (observe only; L_sem + L_dyn alignment)
    Stage 2  — Advisory mode    (non-authoritative; small β_t blending)
    Stage 3  — Co-equal braiding (β_t → 0.5; cross-module co-training)
    Stage 4  — Consolidation    (harden, prune, or factor into solver library)

Fractional distribution
-----------------------
    s_{t,i} = w_q·q_{t,i} + w_u·u_{t,i} + w_r·r_{t,i}
    α_{t,i} = softmax(s_t / τ)_i

    q — competence estimate
    u — epistemic uncertainty
    r — task relevance

Three recursive sync loops
--------------------------
    L_sem  = E_t[ d(P_new(ẑ_new), z_ref) ]          semantic alignment
    L_dyn  = E_t[ d(P_new(ẑ_new), z_wm)  ]          dynamic alignment
    L_pol  = E_t[ KL(π_braid ‖ π_target)  ]          policy alignment

    π_braid = (1-β_t)·π_core + β_t·π_new

Integration with existing stack
--------------------------------
    BraidProjection  — learnable P_i and P_i^{-1}
    ModuleRegistry   — central registry of all registered modules
    FractionalRouter — computes α_t and routes latent flows
    RecursiveSyncLoops — runs L_sem, L_dyn, L_pol optimization steps
    ModuleIntegrationProtocol — orchestrates all stages end-to-end
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ─────────────────────────────────────────────────────────────────────────────
# Module types
# ─────────────────────────────────────────────────────────────────────────────

class ModuleType(Enum):
    PERCEPTION    = "perception"
    MEMORY        = "memory"
    PLANNING      = "planning"
    LANGUAGE      = "language"
    SPATIAL       = "spatial"
    REASONING     = "reasoning"
    META          = "metacognition"
    ACTION        = "action"
    WORLD_MODEL   = "world_model"
    CUSTOM        = "custom"


class IntegrationStage(Enum):
    UNREGISTERED  = 0
    SHADOW        = 1   # observe only
    ADVISORY      = 2   # non-authoritative influence
    CO_EQUAL      = 3   # full braided participation
    CONSOLIDATED  = 4   # hardened / factored
    PRUNED        = 5   # removed from active routing


# ─────────────────────────────────────────────────────────────────────────────
# Module descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModuleDescriptor:
    """
    Formal interface declaration for a cognitive module M_i.

    Every module entering the braid must declare:
        - its type
        - input/output dimensionalities
        - latent dim (z_i space)
        - domain tags (used to initialise relevance r_{t,i})
    """
    name:         str
    module_type:  ModuleType
    input_dim:    int
    output_dim:   int
    latent_dim:   int              # z_i dim before projection to braid space
    domain_tags:  List[str] = field(default_factory=list)
    description:  str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Braid projection  P_i : z_i → z  and  P_i^{-1} : z → z_i
# ─────────────────────────────────────────────────────────────────────────────

class BraidProjection(nn.Module):
    """
    Learnable projections between module latent space z_i and braid space z.

        P_i     : z_i → z     (forward / up-project)
        P_i_inv : z  → z_i    (inverse / down-project)

    Trained during Stage 0 and refined through Stage 1/2 via L_sem + L_dyn.
    """

    def __init__(self, module_latent_dim: int, braid_dim: int, hidden: int = 256):
        super().__init__()
        self.forward_proj = nn.Sequential(
            nn.Linear(module_latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, braid_dim),
        )
        self.inverse_proj = nn.Sequential(
            nn.Linear(braid_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, module_latent_dim),
        )

    def project(self, z_i: torch.Tensor) -> torch.Tensor:
        """P_i(z_i) → z  (braid space)."""
        return self.forward_proj(z_i)

    def unproject(self, z: torch.Tensor) -> torch.Tensor:
        """P_i^{-1}(z) → z_i  (module space)."""
        return self.inverse_proj(z)


# ─────────────────────────────────────────────────────────────────────────────
# Fractional router  — computes α_t over all registered modules
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModuleStats:
    """Per-module running stats used to compute α_{t,i}."""
    competence:   float = 0.1   # q_{t,i} — starts low, earned through performance
    uncertainty:  float = 0.9   # u_{t,i} — starts high
    relevance:    float = 0.5   # r_{t,i} — domain-specific init
    n_updates:    int   = 0


class FractionalRouter(nn.Module):
    """
    Computes the fractional allocation vector α_t over N modules.

        s_{t,i} = w_q·q_{t,i} + w_u·u_{t,i} + w_r·r_{t,i}
        α_{t,i} = softmax(s_t / τ)

    Also handles ε-initialisation for new modules and EMA stat updates.
    """

    def __init__(
        self,
        w_q: float = 0.4,
        w_u: float = 0.3,
        w_r: float = 0.3,
        temperature: float = 1.0,
        eps_new: float = 0.05,
        ema_decay: float = 0.95,
    ):
        super().__init__()
        self.w_q = w_q
        self.w_u = w_u
        self.w_r = w_r
        self.tau = temperature
        self.eps_new = eps_new
        self.ema_decay = ema_decay
        self._stats: Dict[str, ModuleStats] = {}

    def register(self, name: str, init_relevance: float = 0.5) -> None:
        """Register a new module with ε-level initial share."""
        self._stats[name] = ModuleStats(
            competence=self.eps_new,
            uncertainty=1.0 - self.eps_new,
            relevance=init_relevance,
        )

    def update_stats(
        self,
        name: str,
        competence: Optional[float] = None,
        uncertainty: Optional[float] = None,
        relevance: Optional[float] = None,
    ) -> None:
        """EMA update of module stats after an episode."""
        if name not in self._stats:
            self.register(name)
        s = self._stats[name]
        decay = self.ema_decay
        if competence  is not None:
            s.competence  = decay * s.competence  + (1 - decay) * competence
        if uncertainty is not None:
            s.uncertainty = decay * s.uncertainty + (1 - decay) * uncertainty
        if relevance   is not None:
            s.relevance   = decay * s.relevance   + (1 - decay) * relevance
        s.n_updates += 1

    def allocate(
        self,
        active_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute α_t for the given set of active modules (or all registered).

        Returns dict {name: α_{t,i}}.
        """
        names = active_names or list(self._stats.keys())
        if not names:
            return {}

        scores = []
        for n in names:
            s = self._stats.get(n, ModuleStats())
            score = self.w_q * s.competence + self.w_u * s.uncertainty + self.w_r * s.relevance
            scores.append(score)

        scores_t = torch.tensor(scores, dtype=torch.float32)
        alpha = F.softmax(scores_t / self.tau, dim=0)
        return {n: alpha[i].item() for i, n in enumerate(names)}

    def gate_tensor(
        self,
        latents: Dict[str, torch.Tensor],  # {name: [B, D]}
    ) -> torch.Tensor:
        """
        Weighted sum of module latents using current α_t.
        Returns fused braid latent [B, D].
        """
        alpha = self.allocate(list(latents.keys()))
        result = None
        for name, z in latents.items():
            w = alpha.get(name, 0.0)
            result = z * w if result is None else result + z * w
        return result if result is not None else torch.zeros(1)

    @property
    def stats(self) -> Dict[str, ModuleStats]:
        return self._stats


# ─────────────────────────────────────────────────────────────────────────────
# Recursive synchronization losses
# ─────────────────────────────────────────────────────────────────────────────

class RecursiveSyncLoops(nn.Module):
    """
    Three nested braid alignment losses:

        L_sem  — semantic alignment between z̃ and z_ref
        L_dyn  — dynamic alignment between z̃ and z_wm (world-model prediction)
        L_pol  — policy alignment KL(π_braid ‖ π_target)

    Total sync loss:
        L_sync = λ_sem · L_sem + λ_dyn · L_dyn + λ_pol · L_pol
    """

    def __init__(
        self,
        lambda_sem: float = 0.4,
        lambda_dyn: float = 0.4,
        lambda_pol: float = 0.2,
    ):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.lambda_dyn = lambda_dyn
        self.lambda_pol = lambda_pol

    def semantic_loss(
        self,
        z_tilde: torch.Tensor,    # P_new(ẑ_new) reprojected to braid space [B, D]
        z_ref:   torch.Tensor,    # braid reference latent from world-model / ensemble [B, D]
    ) -> torch.Tensor:
        """
        L_sem = E_t[ d(z̃_{t+1}, z_ref_{t+1}) ]
        Using cosine distance as d(·,·) for direction, MSE for magnitude.
        """
        l_mse    = F.mse_loss(z_tilde, z_ref)
        cos_sim  = F.cosine_similarity(z_tilde, z_ref, dim=-1)   # [B]
        l_cosine = (1.0 - cos_sim).mean()
        return 0.5 * l_mse + 0.5 * l_cosine

    def dynamic_loss(
        self,
        z_tilde_next: torch.Tensor,  # P_new(ẑ_{t+1}^new) [B, D]
        z_wm_next:    torch.Tensor,  # world-model prediction z_{t+1}^WM [B, D]
    ) -> torch.Tensor:
        """
        L_dyn = E_t[ d(z̃_{t+1}, z_{t+1}^WM) ]
        MSE in braid space.
        """
        return F.mse_loss(z_tilde_next, z_wm_next)

    def policy_loss(
        self,
        pi_braid_logits:  torch.Tensor,   # [B, A] logits
        pi_target_logits: torch.Tensor,   # [B, A] logits
    ) -> torch.Tensor:
        """
        L_pol = E_t[ KL(π_braid ‖ π_target) ]
        """
        log_braid  = F.log_softmax(pi_braid_logits,  dim=-1)
        log_target = F.log_softmax(pi_target_logits, dim=-1)
        return F.kl_div(log_braid, log_target.exp(), reduction="batchmean")

    def blended_policy(
        self,
        pi_core_logits: torch.Tensor,   # [B, A]
        pi_new_logits:  torch.Tensor,   # [B, A]
        beta: float,                    # blending weight ∈ [0, 1]
    ) -> torch.Tensor:
        """
        π_braid = (1-β)·π_core + β·π_new
        Operates in probability space then converts back to logits.
        """
        p_core = F.softmax(pi_core_logits, dim=-1)
        p_new  = F.softmax(pi_new_logits,  dim=-1)
        p_blend = (1.0 - beta) * p_core + beta * p_new
        return torch.log(p_blend + 1e-8)

    def forward(
        self,
        z_tilde:          Optional[torch.Tensor] = None,
        z_ref:            Optional[torch.Tensor] = None,
        z_tilde_next:     Optional[torch.Tensor] = None,
        z_wm_next:        Optional[torch.Tensor] = None,
        pi_braid_logits:  Optional[torch.Tensor] = None,
        pi_target_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite sync loss. All inputs are optional — missing
        terms are skipped gracefully.

        Returns (L_sync, breakdown_dict).
        """
        device = torch.device("cpu")
        for t in [z_tilde, z_ref, z_tilde_next, z_wm_next,
                  pi_braid_logits, pi_target_logits]:
            if t is not None:
                device = t.device
                break

        zero = torch.zeros(1, device=device)

        # L_sem
        if z_tilde is not None and z_ref is not None:
            l_sem = self.semantic_loss(z_tilde, z_ref)
        else:
            l_sem = zero

        # L_dyn
        if z_tilde_next is not None and z_wm_next is not None:
            l_dyn = self.dynamic_loss(z_tilde_next, z_wm_next)
        else:
            l_dyn = zero

        # L_pol
        if pi_braid_logits is not None and pi_target_logits is not None:
            l_pol = self.policy_loss(pi_braid_logits, pi_target_logits)
        else:
            l_pol = zero

        total = (self.lambda_sem * l_sem
                 + self.lambda_dyn * l_dyn
                 + self.lambda_pol * l_pol)

        breakdown = {
            "l_sem":   l_sem.item(),
            "l_dyn":   l_dyn.item(),
            "l_pol":   l_pol.item(),
            "l_sync":  total.item(),
        }
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Module registry entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegistryEntry:
    descriptor:  ModuleDescriptor
    module:      nn.Module
    projection:  BraidProjection
    stage:       IntegrationStage = IntegrationStage.SHADOW
    beta:        float = 0.0          # policy blending weight
    registered_at: float = field(default_factory=time.time)
    stage_history: List[Tuple[IntegrationStage, float]] = field(default_factory=list)
    metrics:     Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Module Integration Protocol — master orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ModuleIntegrationProtocol(nn.Module):
    """
    Orchestrates the full 5-stage braid integration lifecycle for new modules.

    Stage gates
    -----------
    SHADOW      → ADVISORY      : L_sem < sem_threshold AND L_dyn < dyn_threshold
    ADVISORY    → CO_EQUAL      : avg_reward_delta > advisory_threshold AND
                                  n_advisory_steps > min_advisory_steps
    CO_EQUAL    → CONSOLIDATED  : contribution_score > consolidation_threshold AND
                                  n_coequal_steps > min_coequal_steps
    CO_EQUAL    → PRUNED        : contribution_score < prune_threshold for N steps

    Usage
    -----
        mip = ModuleIntegrationProtocol(braid_dim=256)

        # Register a new module
        desc = ModuleDescriptor("graph_reasoner", ModuleType.REASONING,
                                input_dim=128, output_dim=64, latent_dim=128,
                                domain_tags=["graph", "relational"])
        mip.register(desc, my_graph_module)

        # At each training step:
        sync_loss, breakdown = mip.step(
            name="graph_reasoner",
            z_i=module_latent,
            z_ref=braid_latent,
            z_wm_next=wm_next_latent,
        )
    """

    # Stage promotion thresholds
    SEM_THRESHOLD         = 0.10
    DYN_THRESHOLD         = 0.10
    ADVISORY_REWARD_DELTA = 0.02
    MIN_ADVISORY_STEPS    = 100
    CONSOLIDATION_SCORE   = 0.70
    MIN_COEQUAL_STEPS     = 200
    PRUNE_THRESHOLD       = 0.05
    PRUNE_PATIENCE        = 50   # consecutive steps below threshold before pruning

    # β schedule limits per stage
    BETA_ADVISORY  = 0.15
    BETA_CO_EQUAL  = 0.50

    def __init__(
        self,
        braid_dim: int = 256,
        router_temperature: float = 1.0,
        sync_lambda_sem: float = 0.4,
        sync_lambda_dyn: float = 0.4,
        sync_lambda_pol: float = 0.2,
    ):
        super().__init__()
        self.braid_dim = braid_dim
        self._registry: Dict[str, RegistryEntry] = {}
        self.router  = FractionalRouter(temperature=router_temperature)
        self.sync    = RecursiveSyncLoops(sync_lambda_sem, sync_lambda_dyn, sync_lambda_pol)

        # Step counters per module (stage tracking)
        self._step_counts:   Dict[str, int]   = {}
        self._prune_counters:Dict[str, int]   = {}
        self._reward_history:Dict[str, List[float]] = {}

    # ── Stage 0: Registration ─────────────────────────────────────────────

    def register(
        self,
        descriptor: ModuleDescriptor,
        module: nn.Module,
        init_beta: float = 0.0,
        domain_relevance: float = 0.5,
    ) -> None:
        """
        Stage 0 — Register M_new into the braid.

        Initialises:
        - BraidProjection (P_i and P_i^{-1}) with random weights (train in Stage 1)
        - Fractional router entry with ε-level share
        - RegistryEntry at SHADOW stage
        """
        name = descriptor.name
        proj = BraidProjection(
            module_latent_dim=descriptor.latent_dim,
            braid_dim=self.braid_dim,
        )
        entry = RegistryEntry(
            descriptor=descriptor,
            module=module,
            projection=proj,
            stage=IntegrationStage.SHADOW,
            beta=init_beta,
        )
        entry.stage_history.append((IntegrationStage.SHADOW, time.time()))
        self._registry[name] = entry
        self._step_counts[name]    = 0
        self._prune_counters[name] = 0
        self._reward_history[name] = []

        self.router.register(name, init_relevance=domain_relevance)
        print(f"[MIP] Registered '{name}' ({descriptor.module_type.value}) → Stage 1 SHADOW")

    # ── Core step: run sync loops + update stats ──────────────────────────

    def step(
        self,
        name: str,
        z_i: torch.Tensor,                          # [B, latent_dim] module latent
        z_ref: Optional[torch.Tensor] = None,       # [B, braid_dim]  reference braid latent
        z_wm_next: Optional[torch.Tensor] = None,   # [B, braid_dim]  world-model next latent
        pi_core_logits: Optional[torch.Tensor] = None,
        pi_new_logits:  Optional[torch.Tensor] = None,
        pi_target_logits: Optional[torch.Tensor] = None,
        reward_delta: float = 0.0,
        competence:   float = 0.5,
        uncertainty:  float = 0.5,
        task_relevance: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Single integration step for module `name`.

        Runs the three sync loops appropriate for the current stage,
        updates stats, checks stage promotion criteria.

        Returns (sync_loss, info_dict).
        """
        if name not in self._registry:
            raise KeyError(f"Module '{name}' not registered. Call register() first.")

        entry = self._registry[name]
        stage = entry.stage
        n     = self._step_counts[name]

        # Project z_i → braid space
        z_tilde = entry.projection.project(z_i)         # [B, braid_dim]

        # Compute world-model prediction via inverse + re-project (if available)
        z_tilde_next = None
        if z_wm_next is not None:
            z_i_next      = entry.projection.unproject(z_wm_next)  # [B, latent_dim]
            z_tilde_next  = entry.projection.project(z_i_next)     # [B, braid_dim]

        # Blended policy logits
        pi_braid_logits = None
        if pi_core_logits is not None and pi_new_logits is not None:
            pi_braid_logits = self.sync.blended_policy(
                pi_core_logits, pi_new_logits, beta=entry.beta
            )

        # Run sync loops
        sync_loss, breakdown = self.sync(
            z_tilde=z_tilde           if stage.value >= IntegrationStage.SHADOW.value   else None,
            z_ref=z_ref               if stage.value >= IntegrationStage.SHADOW.value   else None,
            z_tilde_next=z_tilde_next if stage.value >= IntegrationStage.SHADOW.value   else None,
            z_wm_next=z_wm_next       if stage.value >= IntegrationStage.SHADOW.value   else None,
            pi_braid_logits=pi_braid_logits    if stage.value >= IntegrationStage.ADVISORY.value else None,
            pi_target_logits=pi_target_logits  if stage.value >= IntegrationStage.ADVISORY.value else None,
        )

        # Update router stats
        self.router.update_stats(name, competence, uncertainty, task_relevance)
        self._reward_history[name].append(reward_delta)
        self._step_counts[name] += 1
        entry.metrics.update({
            "stage":       stage.value,
            "beta":        entry.beta,
            "n_steps":     n,
            "last_l_sync": sync_loss.item(),
            **breakdown,
        })

        # Stage promotion check
        self._maybe_promote(name, breakdown, reward_delta)

        return sync_loss, entry.metrics

    # ── Stage promotion logic ─────────────────────────────────────────────

    def _maybe_promote(
        self,
        name: str,
        breakdown: Dict[str, float],
        reward_delta: float,
    ) -> None:
        entry = self._registry[name]
        stage = entry.stage
        n     = self._step_counts[name]

        if stage == IntegrationStage.SHADOW:
            # → ADVISORY when semantic + dynamic loss are low enough
            if (breakdown["l_sem"] < self.SEM_THRESHOLD and
                    breakdown["l_dyn"] < self.DYN_THRESHOLD):
                self._advance_stage(name, IntegrationStage.ADVISORY)
                entry.beta = self.BETA_ADVISORY

        elif stage == IntegrationStage.ADVISORY:
            recent_rewards = self._reward_history[name][-20:]
            avg_delta = sum(recent_rewards) / max(len(recent_rewards), 1)
            if (avg_delta > self.ADVISORY_REWARD_DELTA and
                    n > self.MIN_ADVISORY_STEPS):
                self._advance_stage(name, IntegrationStage.CO_EQUAL)
                entry.beta = self.BETA_CO_EQUAL

        elif stage == IntegrationStage.CO_EQUAL:
            alpha = self.router.allocate([name])
            contrib = alpha.get(name, 0.0)
            if contrib > self.CONSOLIDATION_SCORE / 10 and n > self.MIN_COEQUAL_STEPS:
                self._advance_stage(name, IntegrationStage.CONSOLIDATED)
            elif contrib < self.PRUNE_THRESHOLD:
                self._prune_counters[name] += 1
                if self._prune_counters[name] >= self.PRUNE_PATIENCE:
                    self._advance_stage(name, IntegrationStage.PRUNED)
                    print(f"[MIP] '{name}' pruned — contribution below threshold for "
                          f"{self.PRUNE_PATIENCE} steps")
            else:
                self._prune_counters[name] = 0

    def _advance_stage(self, name: str, new_stage: IntegrationStage) -> None:
        entry = self._registry[name]
        old   = entry.stage
        entry.stage = new_stage
        entry.stage_history.append((new_stage, time.time()))
        print(f"[MIP] '{name}' promoted: {old.name} → {new_stage.name}  "
              f"(step {self._step_counts[name]}, β={entry.beta:.3f})")

    # ── Fused braid latent ────────────────────────────────────────────────

    def fused_latent(
        self,
        module_latents: Dict[str, torch.Tensor],  # {name: [B, latent_dim]}
        active_stages: Optional[List[IntegrationStage]] = None,
    ) -> torch.Tensor:
        """
        Compute α-weighted fused braid latent from all active modules.

        Filters to modules in `active_stages` (default: ADVISORY+).
        Returns [B, braid_dim].
        """
        if active_stages is None:
            active_stages = [
                IntegrationStage.ADVISORY,
                IntegrationStage.CO_EQUAL,
                IntegrationStage.CONSOLIDATED,
            ]

        braid_latents: Dict[str, torch.Tensor] = {}
        for name, z_i in module_latents.items():
            entry = self._registry.get(name)
            if entry and entry.stage in active_stages:
                braid_latents[name] = entry.projection.project(z_i)

        if not braid_latents:
            # Fall back to first available latent
            first_name = next(iter(module_latents))
            entry = self._registry[first_name]
            return entry.projection.project(module_latents[first_name])

        return self.router.gate_tensor(braid_latents)

    # ── Diagnostics ───────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return full integration status for all registered modules."""
        alpha = self.router.allocate()
        out = {}
        for name, entry in self._registry.items():
            out[name] = {
                "stage":        entry.stage.name,
                "beta":         round(entry.beta, 4),
                "alpha":        round(alpha.get(name, 0.0), 4),
                "n_steps":      self._step_counts.get(name, 0),
                "metrics":      entry.metrics,
                "module_type":  entry.descriptor.module_type.value,
            }
        return out

    def active_modules(self) -> List[str]:
        return [
            n for n, e in self._registry.items()
            if e.stage not in (IntegrationStage.PRUNED, IntegrationStage.UNREGISTERED)
        ]
