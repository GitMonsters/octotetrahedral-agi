"""
Recursive Engine Objective Function
=====================================
Implements the composite multi-objective loss for the Recursive Engine,
optimized for general adaptability rather than benchmark memorization.

Primary utility objective
-------------------------
    J = E_{e~E}[U(e) - λ1·C(e) - λ2·R(e) + λ3·A(e) + λ4·M(e)]

Where:
    U(e)  — task utility / goal achievement
    C(e)  — compute, energy, and latency cost
    R(e)  — risk of unsafe, brittle, or irreversible actions
    A(e)  — adaptability gain under novelty and distribution shift
    M(e)  — model improvement from experience (self-correction, abstraction)

Training loss (six-term composite)
-----------------------------------
    L_total = L_task
            + λ1·L_WM          (world-model prediction)
            + λ2·L_meta        (meta-learning efficiency)
            + λ3·L_resource    (compute proportionality)
            + λ4·L_ground      (embodied grounding)
            + λ5·L_stability   (coherence / anti-forgetting)

Design principle
----------------
    Outer: maximise long-run general adaptability
    Inner: solve the current task efficiently and safely

This module is pure PyTorch — no dependency on heavy ARC-specific code.
It can be dropped into any training loop that provides the required tensors.

Integration hooks
-----------------
    AdaptiveComputationController  →  ponder_cost → L_resource
    CognitiveCohesionBraid         →  cohesion_score → L_stability
    FluidIntelligenceEngine        →  adaptation_delta → L_meta
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
class RecursiveEngineConfig:
    """Weights and hyper-parameters for the composite objective."""

    # Primary utility weights (J)
    lambda_cost: float = 0.05          # λ1 — penalise unnecessary compute
    lambda_risk: float = 0.10          # λ2 — penalise unsafe / brittle actions
    lambda_adapt: float = 0.15         # λ3 — reward adaptability gain
    lambda_model_improvement: float = 0.10  # λ4 — reward self-correction

    # Training loss weights
    lambda_wm: float = 0.20            # λ1 — world-model loss
    lambda_meta: float = 0.15          # λ2 — meta-learning loss
    lambda_resource: float = 0.05      # λ3 — resource waste penalty
    lambda_ground: float = 0.10        # λ4 — grounding mismatch
    lambda_stability: float = 0.15     # λ5 — incoherence / forgetting

    # World-model sub-weights (L_WM = α·L_pred + β·L_causal + γ·L_rollout + δ·L_calib)
    wm_alpha: float = 0.40             # next-step prediction
    wm_beta: float = 0.25              # causal intervention accuracy
    wm_gamma: float = 0.25             # long-horizon rollout consistency
    wm_delta: float = 0.10             # uncertainty calibration

    # Resource target (ponder cost should converge here)
    resource_target_cost: float = 0.50

    # Stability — target cohesion score (higher = more aligned)
    stability_target_cohesion: float = 0.80

    # Meta-learning history window for computing ΔAdaptationTime
    meta_window: int = 32

    # Uncertainty threshold: above this, deeper inference is rewarded
    uncertainty_threshold: float = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# World-Model loss  (L_WM)
# ─────────────────────────────────────────────────────────────────────────────

class WorldModelLoss(nn.Module):
    """
    L_WM = α·L_pred + β·L_causal + γ·L_rollout + δ·L_calib

    Prevents superficial pattern matching by requiring:
    - accurate next-step predictions
    - causal intervention accuracy
    - long-horizon rollout consistency
    - calibrated uncertainty estimates
    """

    def __init__(self, config: RecursiveEngineConfig):
        super().__init__()
        self.cfg = config

    def forward(
        self,
        pred_next: torch.Tensor,       # [B, D] predicted next state
        true_next: torch.Tensor,       # [B, D] actual next state
        pred_causal: Optional[torch.Tensor] = None,   # [B, D] predicted post-intervention state
        true_causal: Optional[torch.Tensor] = None,   # [B, D] actual post-intervention state
        rollout_preds: Optional[torch.Tensor] = None, # [B, T, D] multi-step rollout predictions
        rollout_trues: Optional[torch.Tensor] = None, # [B, T, D] multi-step ground truth
        uncertainty: Optional[torch.Tensor] = None,   # [B, D] predicted uncertainty (log-variance)
        true_variance: Optional[torch.Tensor] = None, # [B, D] empirical variance
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg

        # L_pred: immediate next-step prediction
        l_pred = F.mse_loss(pred_next, true_next)

        # L_causal: counterfactual / intervention accuracy
        if pred_causal is not None and true_causal is not None:
            l_causal = F.mse_loss(pred_causal, true_causal)
        else:
            l_causal = torch.zeros(1, device=pred_next.device)

        # L_rollout: long-horizon consistency (average over time steps)
        if rollout_preds is not None and rollout_trues is not None:
            # Weight later time-steps more heavily — those require better models
            T = rollout_preds.size(1)
            weights = torch.arange(1, T + 1, dtype=torch.float32, device=pred_next.device)
            weights = weights / weights.sum()
            step_losses = F.mse_loss(rollout_preds, rollout_trues, reduction='none').mean(-1)  # [B, T]
            l_rollout = (step_losses * weights.unsqueeze(0)).sum(-1).mean()
        else:
            l_rollout = torch.zeros(1, device=pred_next.device)

        # L_calib: uncertainty calibration via negative log-likelihood (NLL)
        if uncertainty is not None and true_variance is not None:
            # Gaussian NLL: 0.5 * (log_var + (true_var / exp(log_var)))
            log_var = uncertainty.clamp(-10, 10)
            l_calib = 0.5 * (log_var + true_variance / log_var.exp()).mean()
        else:
            l_calib = torch.zeros(1, device=pred_next.device)

        total = (cfg.wm_alpha * l_pred
                 + cfg.wm_beta  * l_causal
                 + cfg.wm_gamma * l_rollout
                 + cfg.wm_delta * l_calib)

        breakdown = {
            "l_pred":    l_pred.item(),
            "l_causal":  l_causal.item(),
            "l_rollout": l_rollout.item(),
            "l_calib":   l_calib.item(),
        }
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Meta-learning loss  (L_meta)
# ─────────────────────────────────────────────────────────────────────────────

class MetaLearningLoss(nn.Module):
    """
    L_meta = -ΔAdaptationTime - ΔErrorAfterShift

    Rewards:
    - faster adaptation to new task structure
    - lower error after a distribution shift
    - better hypothesis generation
    - fewer updates needed for competence

    Uses an exponential moving average window to compute deltas, so the
    loss can be computed online without storing full history.
    """

    def __init__(self, config: RecursiveEngineConfig):
        super().__init__()
        self.cfg = config
        self._ema_adaptation_time: float = 1.0   # steps to competence
        self._ema_error_after_shift: float = 1.0  # error on first post-shift batch
        self._decay = 2.0 / (config.meta_window + 1)

    def update_and_compute(
        self,
        current_adaptation_time: float,
        current_error_after_shift: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Update EMA and return the meta-learning loss.

        Args:
            current_adaptation_time:  steps taken to reach competence on this task
            current_error_after_shift: task error on first evaluation after distribution shift
        """
        prev_adapt = self._ema_adaptation_time
        prev_error = self._ema_error_after_shift

        # EMA update
        self._ema_adaptation_time = (
            self._decay * current_adaptation_time
            + (1 - self._decay) * self._ema_adaptation_time
        )
        self._ema_error_after_shift = (
            self._decay * current_error_after_shift
            + (1 - self._decay) * self._ema_error_after_shift
        )

        # Positive delta means things got worse → penalise
        delta_adapt = self._ema_adaptation_time - prev_adapt
        delta_error = self._ema_error_after_shift - prev_error

        # L_meta = -ΔAdaptationTime - ΔErrorAfterShift
        # (minimising this maximises improvement in both quantities)
        loss = torch.tensor(delta_adapt + delta_error, dtype=torch.float32)

        breakdown = {
            "ema_adaptation_time":    round(self._ema_adaptation_time, 4),
            "ema_error_after_shift":  round(self._ema_error_after_shift, 4),
            "delta_adapt":            round(delta_adapt, 4),
            "delta_error":            round(delta_error, 4),
        }
        return loss, breakdown

    def forward(
        self,
        current_adaptation_time: float,
        current_error_after_shift: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.update_and_compute(current_adaptation_time, current_error_after_shift)


# ─────────────────────────────────────────────────────────────────────────────
# Resource loss  (L_resource)
# ─────────────────────────────────────────────────────────────────────────────

class ResourceLoss(nn.Module):
    """
    L_resource = ComputeUsed × f(TaskDifficulty, Uncertainty)

    The system should spend its attention budget proportionally to
    epistemic uncertainty, not uniformly:
    - Easy / low-uncertainty tasks → penalise over-compute
    - Hard / high-uncertainty tasks → allow deeper inference

    f(d, u) is a scaling function that rewards proportionality.
    When ComputeUsed is exactly right given difficulty+uncertainty,
    this loss approaches zero.
    """

    def __init__(self, config: RecursiveEngineConfig):
        super().__init__()
        self.cfg = config

    def forward(
        self,
        ponder_cost: torch.Tensor,      # [B] actual compute used (0..1 from ACT)
        task_difficulty: torch.Tensor,   # [B] estimated task difficulty (0..1)
        uncertainty: torch.Tensor,       # [B] epistemic uncertainty (0..1)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Ideal compute = f(difficulty, uncertainty)
        # Blend: high difficulty + high uncertainty → more compute is appropriate
        ideal_compute = 0.5 * task_difficulty + 0.5 * uncertainty  # [B]
        ideal_compute = ideal_compute.clamp(0.0, 1.0)

        # Penalty: squared deviation from ideal
        # Over-compute on easy tasks: positive deviation, penalised
        # Under-compute on hard tasks: negative deviation, penalised equally
        deviation = ponder_cost - ideal_compute
        l_resource = (deviation ** 2).mean()

        breakdown = {
            "mean_ponder_cost":   ponder_cost.mean().item(),
            "mean_ideal_compute": ideal_compute.mean().item(),
            "mean_deviation":     deviation.mean().item(),
        }
        return l_resource, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Grounding loss  (L_ground)
# ─────────────────────────────────────────────────────────────────────────────

class GroundingLoss(nn.Module):
    """
    L_ground = mismatch between internal predictions and external feedback.

    Ties the braid to reality instead of allowing self-consistent abstraction.
    Rewards:
    - action-outcome accuracy
    - perception-action consistency
    - sensorimotor alignment
    - correction from real feedback
    """

    def __init__(self, config: RecursiveEngineConfig):
        super().__init__()
        self.cfg = config

    def forward(
        self,
        predicted_outcome: torch.Tensor,   # [B, D] what the model expected to happen
        actual_outcome: torch.Tensor,      # [B, D] what actually happened
        predicted_perception: Optional[torch.Tensor] = None,  # [B, P] predicted sensory state
        actual_perception: Optional[torch.Tensor] = None,     # [B, P] actual sensory state
        action_consistency: Optional[torch.Tensor] = None,    # [B] 0/1 per sample
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Action-outcome accuracy
        l_outcome = F.mse_loss(predicted_outcome, actual_outcome)

        # Perception-action consistency
        if predicted_perception is not None and actual_perception is not None:
            l_perception = F.mse_loss(predicted_perception, actual_perception)
        else:
            l_perception = torch.zeros(1, device=predicted_outcome.device)

        # Sensorimotor alignment bonus (reward when actions match sensory consequences)
        if action_consistency is not None:
            # 1 - mean consistency means we minimise inconsistency
            l_sensorimotor = 1.0 - action_consistency.float().mean()
        else:
            l_sensorimotor = torch.zeros(1, device=predicted_outcome.device)

        total = (0.50 * l_outcome + 0.30 * l_perception + 0.20 * l_sensorimotor)

        breakdown = {
            "l_outcome":      l_outcome.item(),
            "l_perception":   l_perception.item(),
            "l_sensorimotor": l_sensorimotor.item(),
        }
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Stability loss  (L_stability)
# ─────────────────────────────────────────────────────────────────────────────

class StabilityLoss(nn.Module):
    """
    L_stability = penalty for incoherence, forgetting, or oscillation.

    Sources:
    1. Cohesion deficit from CognitiveCohesionBraid (1 - cohesion_score)
    2. Parameter drift (EWC-style forgetting penalty)
    3. Output oscillation across consecutive steps

    Penalises:
    - catastrophic forgetting
    - oscillatory reasoning
    - fragmentation across modules
    """

    def __init__(self, config: RecursiveEngineConfig):
        super().__init__()
        self.cfg = config
        # Fisher information matrices for EWC (populated during consolidation)
        self._fisher: Dict[str, torch.Tensor] = {}
        self._anchors: Dict[str, torch.Tensor] = {}

    def consolidate(self, named_params: Dict[str, torch.Tensor]):
        """
        Call after completing a task family to anchor important parameters.
        Computes diagonal Fisher approximation from stored gradients.
        """
        for name, param in named_params.items():
            if param.grad is not None:
                self._fisher[name] = param.grad.data.clone().pow(2)
                self._anchors[name] = param.data.clone()

    def forward(
        self,
        named_params: Optional[Dict[str, torch.Tensor]] = None,
        ref_params: Optional[Dict[str, torch.Tensor]] = None,
        cohesion_score: float = 1.0,
        prev_output: Optional[torch.Tensor] = None,
        curr_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes stability loss.

        Formal spec
        -----------
        L_stab = κ1·D(φ, φ_ref) + κ2·D(θ, θ_ref)

        where D is squared L2 distance and φ_ref/θ_ref are EMA reference
        weights.  EWC forgetting penalty (diagonal Fisher) is used as a
        complementary term when ref_params is not provided.

        Args
        ----
        named_params    : current model params {name: tensor}
        ref_params      : EMA reference params {name: tensor}  (formal spec)
        cohesion_score  : float from CognitiveCohesionBraid
        prev_output     : previous step latent/output (oscillation detection)
        curr_output     : current step latent/output
        """
        device = torch.device("cpu")
        if named_params:
            first_param = next(iter(named_params.values()), None)
            if first_param is not None:
                device = first_param.device
        elif prev_output is not None:
            device = prev_output.device

        # 1. Cohesion deficit
        l_cohesion = torch.tensor(
            max(0.0, self.cfg.stability_target_cohesion - cohesion_score),
            dtype=torch.float32, device=device
        )

        # 2a. EMA-reference distance  D(φ, φ_ref)  — formal spec L_stab
        l_forgetting = torch.zeros(1, device=device)
        if named_params and ref_params:
            n = 0
            for name, param in named_params.items():
                if name in ref_params:
                    ref_p = ref_params[name].to(device).detach()
                    l_forgetting = l_forgetting + F.mse_loss(param, ref_p)
                    n += 1
            if n > 0:
                l_forgetting = l_forgetting / n
        elif named_params and self._fisher:
            # Fallback: EWC forgetting penalty when no EMA reference provided
            for name, param in named_params.items():
                if name in self._fisher and name in self._anchors:
                    fisher = self._fisher[name].to(device)
                    anchor = self._anchors[name].to(device)
                    l_forgetting = l_forgetting + (fisher * (param - anchor).pow(2)).sum()
            l_forgetting = l_forgetting / max(len(self._fisher), 1)

        # 3. Oscillation penalty
        if prev_output is not None and curr_output is not None:
            # High cosine similarity reversal between steps = oscillation
            prev_norm = F.normalize(prev_output.float(), dim=-1)
            curr_norm = F.normalize(curr_output.float(), dim=-1)
            cosine_sim = (prev_norm * curr_norm).sum(-1)  # [B]
            # Penalise large direction reversals (cosine_sim < 0)
            l_oscillation = F.relu(-cosine_sim).mean()
        else:
            l_oscillation = torch.zeros(1, device=device)

        total = 0.40 * l_cohesion + 0.40 * l_forgetting + 0.20 * l_oscillation

        breakdown = {
            "cohesion_score":    round(cohesion_score, 4),
            "l_cohesion":        l_cohesion.item(),
            "l_forgetting":      l_forgetting.item(),
            "l_oscillation":     l_oscillation.item(),
        }
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Master composite objective
# ─────────────────────────────────────────────────────────────────────────────

class RecursiveEngineObjective(nn.Module):
    """
    Master composite objective for the Recursive Engine.

    Two-level design:
        Outer: maximise long-run general adaptability  (J utility objective)
        Inner: solve the current task efficiently/safely (L_total training loss)

    Usage
    -----
        obj = RecursiveEngineObjective(config)

        # During each training step:
        loss, metrics = obj.compute_loss(
            task_loss          = cross_entropy_or_mse,
            pred_next          = model_next_state_pred,
            true_next          = actual_next_state,
            adaptation_time    = steps_to_solve,
            error_after_shift  = post_shift_error,
            ponder_cost        = act_controller.get_ponder_cost(),
            task_difficulty    = difficulty_estimate,
            uncertainty        = model_uncertainty,
            predicted_outcome  = model_outcome_pred,
            actual_outcome     = real_outcome,
            cohesion_score     = braid.cohesion_score()["cohesion_score"],
        )

        loss.backward()

    Periodically (e.g., after each task family):
        obj.stability_loss.consolidate(dict(model.named_parameters()))
    """

    def __init__(self, config: Optional[RecursiveEngineConfig] = None):
        super().__init__()
        self.cfg = config or RecursiveEngineConfig()
        self.world_model_loss = WorldModelLoss(self.cfg)
        self.meta_loss        = MetaLearningLoss(self.cfg)
        self.resource_loss    = ResourceLoss(self.cfg)
        self.grounding_loss   = GroundingLoss(self.cfg)
        self.stability_loss   = StabilityLoss(self.cfg)

    def compute_utility(
        self,
        task_utility: float,
        compute_cost: float,
        risk: float,
        adaptability_gain: float,
        model_improvement: float,
    ) -> float:
        """
        J = U(e) - λ1·C(e) - λ2·R(e) + λ3·A(e) + λ4·M(e)

        Scalar utility estimate for logging / RL outer-loop.
        """
        cfg = self.cfg
        return (
            task_utility
            - cfg.lambda_cost  * compute_cost
            - cfg.lambda_risk  * risk
            + cfg.lambda_adapt * adaptability_gain
            + cfg.lambda_model_improvement * model_improvement
        )

    def compute_loss(
        self,
        # ── Inner task loss ──────────────────────────────────────────────────
        task_loss: torch.Tensor,
        # ── World-model tensors ─────────────────────────────────────────────
        pred_next: Optional[torch.Tensor] = None,
        true_next: Optional[torch.Tensor] = None,
        pred_causal: Optional[torch.Tensor] = None,
        true_causal: Optional[torch.Tensor] = None,
        rollout_preds: Optional[torch.Tensor] = None,
        rollout_trues: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        true_variance: Optional[torch.Tensor] = None,
        # ── Meta-learning scalars ───────────────────────────────────────────
        adaptation_time: Optional[float] = None,
        error_after_shift: Optional[float] = None,
        # ── Resource tensors ────────────────────────────────────────────────
        ponder_cost: Optional[torch.Tensor] = None,
        task_difficulty: Optional[torch.Tensor] = None,
        # ── Grounding tensors ───────────────────────────────────────────────
        predicted_outcome: Optional[torch.Tensor] = None,
        actual_outcome: Optional[torch.Tensor] = None,
        predicted_perception: Optional[torch.Tensor] = None,
        actual_perception: Optional[torch.Tensor] = None,
        action_consistency: Optional[torch.Tensor] = None,
        # ── Stability inputs ────────────────────────────────────────────────
        named_params: Optional[Dict[str, torch.Tensor]] = None,
        cohesion_score: float = 1.0,
        prev_output: Optional[torch.Tensor] = None,
        curr_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute L_total and a metrics dict for logging.

        Returns
        -------
        total_loss  : scalar Tensor (backpropagatable)
        metrics     : dict with per-term losses and breakdowns
        """
        device = task_loss.device
        total = task_loss.clone()
        metrics: Dict[str, float] = {"l_task": task_loss.item()}

        # World-model loss
        if pred_next is not None and true_next is not None:
            l_wm, wm_breakdown = self.world_model_loss(
                pred_next, true_next,
                pred_causal, true_causal,
                rollout_preds, rollout_trues,
                uncertainty, true_variance,
            )
            total = total + self.cfg.lambda_wm * l_wm
            metrics["l_wm"] = l_wm.item()
            metrics.update({f"wm_{k}": v for k, v in wm_breakdown.items()})

        # Meta-learning loss
        if adaptation_time is not None and error_after_shift is not None:
            l_meta, meta_breakdown = self.meta_loss(adaptation_time, error_after_shift)
            l_meta = l_meta.to(device)
            total = total + self.cfg.lambda_meta * l_meta
            metrics["l_meta"] = l_meta.item()
            metrics.update({f"meta_{k}": v for k, v in meta_breakdown.items()})

        # Resource / compute loss
        if ponder_cost is not None and task_difficulty is not None and uncertainty is not None:
            l_res, res_breakdown = self.resource_loss(
                ponder_cost.to(device),
                task_difficulty.to(device),
                uncertainty.to(device),
            )
            total = total + self.cfg.lambda_resource * l_res
            metrics["l_resource"] = l_res.item()
            metrics.update({f"res_{k}": v for k, v in res_breakdown.items()})

        # Grounding loss
        if predicted_outcome is not None and actual_outcome is not None:
            l_ground, ground_breakdown = self.grounding_loss(
                predicted_outcome, actual_outcome,
                predicted_perception, actual_perception,
                action_consistency,
            )
            total = total + self.cfg.lambda_ground * l_ground
            metrics["l_ground"] = l_ground.item()
            metrics.update({f"ground_{k}": v for k, v in ground_breakdown.items()})

        # Stability loss
        l_stab, stab_breakdown = self.stability_loss(
            named_params=named_params,
            cohesion_score=cohesion_score,
            prev_output=prev_output,
            curr_output=curr_output,
        )
        l_stab = l_stab.to(device)
        total = total + self.cfg.lambda_stability * l_stab
        metrics["l_stability"] = l_stab.item()
        metrics.update({f"stab_{k}": v for k, v in stab_breakdown.items()})

        metrics["l_total"] = total.item()
        return total, metrics

    def consolidate_after_task(self, model: nn.Module):
        """
        Call after completing a task family to anchor EWC parameters.
        Prevents catastrophic forgetting across task distributions.
        """
        self.stability_loss.consolidate(
            {n: p for n, p in model.named_parameters() if p.requires_grad}
        )

    def log_priority_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Map composite metrics to the six objective priorities
        (for dashboards / wandb logging):

            1. Generalization under novelty  →  meta_delta_error
            2. Prediction quality            →  wm_l_pred + wm_l_rollout
            3. Sample/compute efficiency     →  res_mean_deviation
            4. Self-correction               →  meta_delta_adapt
            5. Stability & coherence         →  stab_cohesion_score
            6. Embodied alignment            →  ground_l_outcome
        """
        return {
            "priority_generalization":  metrics.get("meta_delta_error", 0.0),
            "priority_prediction":      metrics.get("wm_l_pred", 0.0)
                                        + metrics.get("wm_l_rollout", 0.0),
            "priority_efficiency":      abs(metrics.get("res_mean_deviation", 0.0)),
            "priority_self_correction": metrics.get("meta_delta_adapt", 0.0),
            "priority_stability":       metrics.get("stab_cohesion_score", 1.0),
            "priority_grounding":       metrics.get("ground_l_outcome", 0.0),
        }
