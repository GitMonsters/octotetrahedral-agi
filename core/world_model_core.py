"""
World Model Core  (W_φ)
========================
Implements the active world-model pillar of the Recursive Engine.

Formal spec
-----------
Given environment e with POMDP  M_e = (S_e, A_e, P_e, R_e, Ω_e, O_e):

    Latent dynamics   :  p_φ(z_{t+1} | z_t, a_t)   — predict next latent
    Observation model :  p_φ(o_t     | z_t)          — decode latent → obs
    Reward model      :  r_φ(z_t, a_t)               — predict reward
    Encoder           :  q_φ(z_t     | h_t)           — history → latent

World-model loss (per formal spec)
-----------------------------------
    L_WM = α·L_pred + β·L_causal + γ·L_rollout + δ·L_calib

    L_pred    = Σ_t E[-log p_φ(o_t | z_t)]
    L_causal  = KL(p_φ(z' | do(a)) ‖ p_real(z' | do(a)))
    L_rollout = discrepancy between imagined rollout z_{t+k} and inferred latents
    L_calib   = calibration error between predicted and empirical uncertainty

Architecture
------------
    HistoryEncoder      →  latent z_t    (GRU over (o, a, r) history)
    LatentDynamics      →  z_{t+1}       (MLP with Gaussian head)
    ObservationDecoder  →  p(o | z)      (MLP with Gaussian head)
    RewardPredictor     →  r̂(z, a)       (MLP scalar)
    UncertaintyEstimator→  u(z)          (ensemble disagreement or learned)
    WorldModel          →  orchestrator  (all of the above)

Integration
-----------
    Used by RecursiveEngineTrainer for:
    - plan_with_world_model(budget=c_t)
    - compute_world_model_loss(D)
    - compute_uncertainty(h) → feeds ResourceAllocator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorldModelConfig:
    obs_dim: int   = 128   # dimensionality of observations o_t
    act_dim: int   = 32    # dimensionality of actions a_t
    latent_dim: int = 256  # dimensionality of latent state z_t
    hidden_dim: int = 512  # hidden layer width
    rnn_layers: int = 1    # GRU layers for history encoder

    # Rollout planning
    max_rollout_steps: int = 8

    # Loss weights  (L_WM = α·L_pred + β·L_causal + γ·L_rollout + δ·L_calib)
    alpha: float = 0.40   # prediction
    beta:  float = 0.25   # causal
    gamma: float = 0.25   # rollout
    delta: float = 0.10   # calibration

    # KL balancing for latent dynamics (like DreamerV2)
    kl_free_nats: float = 1.0
    kl_balance: float = 0.8   # weight of stop-grad posterior in KL

    # Ensemble size for uncertainty estimation
    ensemble_size: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class HistoryEncoder(nn.Module):
    """
    q_φ(z_t | h_t) — encodes observation/action/reward history to latent.

    h_t = (o_1, a_1, r_1, ..., o_t)
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.obs_dim + cfg.act_dim + 1  # +1 for reward scalar

        self.embed = nn.Linear(input_dim, cfg.hidden_dim)
        self.rnn = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.rnn_layers,
            batch_first=True,
        )
        self.mu_head    = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.sigma_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(
        self,
        obs: torch.Tensor,     # [B, T, obs_dim]
        actions: torch.Tensor, # [B, T, act_dim]
        rewards: torch.Tensor, # [B, T, 1]
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_mu     [B, latent_dim]
            z_sigma  [B, latent_dim]  (positive)
            rnn_hidden  for continuation
        """
        x = torch.cat([obs, actions, rewards], dim=-1)       # [B, T, obs+act+1]
        x = F.silu(self.embed(x))                            # [B, T, hidden]
        out, hidden = self.rnn(x, hidden)                    # [B, T, hidden]
        last = out[:, -1]                                    # [B, hidden]
        mu    = self.mu_head(last)
        sigma = F.softplus(self.sigma_head(last)) + 1e-5
        return mu, sigma, hidden


class LatentDynamics(nn.Module):
    """
    p_φ(z_{t+1} | z_t, a_t) — deterministic + stochastic latent transition.
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.act_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.mu_head    = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.sigma_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

    def forward(
        self,
        z: torch.Tensor,   # [B, latent_dim]
        a: torch.Tensor,   # [B, act_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, a], dim=-1)
        h = self.net(x)
        mu    = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-5
        return mu, sigma


class ObservationDecoder(nn.Module):
    """
    p_φ(o_t | z_t) — reconstruct/predict observation from latent.
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.mu_head    = nn.Linear(cfg.hidden_dim, cfg.obs_dim)
        self.sigma_head = nn.Linear(cfg.hidden_dim, cfg.obs_dim)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z)
        mu    = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-5
        return mu, sigma


class RewardPredictor(nn.Module):
    """
    r_φ(z_t, a_t) — predict scalar reward from latent + action.
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.act_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)  # [B]


class UncertaintyEstimator(nn.Module):
    """
    Ensemble-based epistemic uncertainty u(z_t).

    Trains `ensemble_size` prediction heads; disagreement = uncertainty.
    Higher uncertainty → ResourceAllocator allocates more compute.
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.latent_dim, cfg.hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim // 2, cfg.latent_dim),
            )
            for _ in range(cfg.ensemble_size)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_pred   [B, latent_dim] — ensemble mean
            uncertainty [B]             — scalar epistemic uncertainty (0..1)
        """
        preds = torch.stack([h(z) for h in self.heads], dim=0)  # [E, B, D]
        mean_pred   = preds.mean(dim=0)                          # [B, D]
        variance    = preds.var(dim=0).mean(dim=-1)              # [B]
        # Normalise to ~(0, 1) via sigmoid of log-variance
        uncertainty = torch.sigmoid(torch.log(variance + 1e-8))  # [B]
        return mean_pred, uncertainty


# ─────────────────────────────────────────────────────────────────────────────
# World Model orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    W_φ — full world model combining all sub-modules.

    Key methods
    -----------
    encode(obs, actions, rewards)   → z_mu, z_sigma
    imagine(z, actions)             → rollout of latent states
    decode(z)                       → observation prediction
    predict_reward(z, a)            → scalar reward
    uncertainty(z)                  → epistemic uncertainty score
    compute_loss(batch)             → L_WM scalar + breakdown dict
    """

    def __init__(self, cfg: Optional[WorldModelConfig] = None):
        super().__init__()
        self.cfg = cfg or WorldModelConfig()
        self.encoder    = HistoryEncoder(self.cfg)
        self.dynamics   = LatentDynamics(self.cfg)
        self.decoder    = ObservationDecoder(self.cfg)
        self.reward_net = RewardPredictor(self.cfg)
        self.uncertainty_estimator = UncertaintyEstimator(self.cfg)

    # ── Core forward passes ─────────────────────────────────────────────────

    def encode(
        self,
        obs: torch.Tensor,      # [B, T, obs_dim]
        actions: torch.Tensor,  # [B, T, act_dim]
        rewards: torch.Tensor,  # [B, T, 1]
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q_φ(z_t | h_t) → z_mu, z_sigma, rnn_hidden."""
        return self.encoder(obs, actions, rewards, hidden)

    def sample_latent(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterised sample from q_φ."""
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def imagine(
        self,
        z0: torch.Tensor,       # [B, latent_dim] initial latent
        actions: torch.Tensor,  # [B, T, act_dim] planned action sequence
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step latent rollout under the dynamics model.

        Returns:
            z_seq   [B, T+1, latent_dim]  — latent trajectory (incl. z0)
            u_seq   [B, T+1]              — uncertainty at each step
        """
        T = actions.size(1)
        z = z0
        z_seq = [z.unsqueeze(1)]
        u_seq = []

        _, u0 = self.uncertainty_estimator(z)
        u_seq.append(u0.unsqueeze(1))

        for t in range(T):
            a_t = actions[:, t]
            mu, sigma = self.dynamics(z, a_t)
            z = self.sample_latent(mu, sigma)
            z_seq.append(z.unsqueeze(1))
            _, u_t = self.uncertainty_estimator(z)
            u_seq.append(u_t.unsqueeze(1))

        return torch.cat(z_seq, dim=1), torch.cat(u_seq, dim=1)  # [B, T+1, D], [B, T+1]

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """p_φ(o | z) → mu, sigma."""
        return self.decoder(z)

    def predict_reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """r_φ(z, a) → scalar reward."""
        return self.reward_net(z, a)

    def compute_uncertainty(self, z: torch.Tensor) -> torch.Tensor:
        """Returns scalar uncertainty per sample: [B]."""
        _, u = self.uncertainty_estimator(z)
        return u

    # ── Loss computation ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        obs: torch.Tensor,              # [B, T, obs_dim]
        actions: torch.Tensor,          # [B, T, act_dim]
        rewards: torch.Tensor,          # [B, T, 1]
        next_obs: Optional[torch.Tensor] = None,   # [B, T, obs_dim]
        causal_obs: Optional[torch.Tensor] = None, # [B, D] post-intervention obs
        causal_pred: Optional[torch.Tensor] = None,# [B, D] predicted post-intervention
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        L_WM = α·L_pred + β·L_causal + γ·L_rollout + δ·L_calib

        Returns (loss scalar, breakdown dict).
        """
        cfg = self.cfg
        B, T, _ = obs.shape
        device = obs.device

        # Encode history to get posterior latent
        z_mu, z_sigma, _ = self.encode(obs, actions, rewards)  # [B, D]
        z = self.sample_latent(z_mu, z_sigma)                  # [B, D]

        # ── L_pred: Σ_t E[-log p_φ(o_t | z_t)] ─────────────────────────────
        # Reconstruct each observation from its corresponding latent
        # For efficiency: encode step-by-step (simplified: use current z for all)
        obs_mu, obs_sigma = self.decode(z)  # [B, obs_dim]
        target = obs[:, -1] if next_obs is None else next_obs[:, -1]
        dist_pred = Normal(obs_mu, obs_sigma)
        l_pred = -dist_pred.log_prob(target).mean()

        # ── L_causal: KL between predicted and realized post-intervention ────
        if causal_pred is not None and causal_obs is not None:
            # Predict post-intervention latent via dynamics
            a_dummy = torch.zeros(B, cfg.act_dim, device=device)
            c_mu, c_sigma = self.dynamics(z, a_dummy)
            dist_pred_c = Normal(c_mu, c_sigma)
            # Realised: encode intervention observation
            causal_z = causal_pred  # already latent
            l_causal = F.mse_loss(c_mu, causal_z)
        else:
            l_causal = torch.zeros(1, device=device)

        # ── L_rollout: imagined vs inferred latent multi-step ────────────────
        z_seq, _ = self.imagine(z, actions)  # [B, T+1, D]
        imagined = z_seq[:, 1:]              # [B, T, D]
        # Re-encode each step's obs as reference latent
        with torch.no_grad():
            all_mu = []
            for t in range(T):
                o_t = obs[:, :t+1]
                a_t = actions[:, :t+1]
                r_t = rewards[:, :t+1]
                m, _, _ = self.encode(o_t, a_t, r_t)
                all_mu.append(m.unsqueeze(1))
            inferred = torch.cat(all_mu, dim=1)  # [B, T, D]

        # Time-weighted: later steps penalised more
        weights = torch.arange(1, T + 1, dtype=torch.float32, device=device) / T
        step_mse = F.mse_loss(imagined, inferred, reduction='none').mean(-1)  # [B, T]
        l_rollout = (step_mse * weights.unsqueeze(0)).sum(-1).mean()

        # ── L_calib: calibration between predicted σ and empirical error ─────
        empirical_err = (obs_mu.detach() - target).pow(2)  # [B, obs_dim]
        l_calib = F.mse_loss(obs_sigma.pow(2), empirical_err)

        # ── Composite ────────────────────────────────────────────────────────
        total = (cfg.alpha * l_pred
                 + cfg.beta  * l_causal
                 + cfg.gamma * l_rollout
                 + cfg.delta * l_calib)

        breakdown = {
            "l_pred":    l_pred.item(),
            "l_causal":  l_causal.item(),
            "l_rollout": l_rollout.item(),
            "l_calib":   l_calib.item(),
            "l_wm_total": total.item(),
        }
        return total, breakdown
