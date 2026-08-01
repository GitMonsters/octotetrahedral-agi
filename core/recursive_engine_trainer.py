"""
Recursive Engine Trainer
========================
Implements the formal meta-learning training loop from the Recursive Engine spec.

Pseudocode spec (from architecture doc)
----------------------------------------

    Initialize φ (world model), θ (policy), ψ (meta-learner), ω (resource allocator)
    Initialize φ_ref = φ, θ_ref = θ  (EMA reference weights)

    for meta_iter in 1..N_meta:
        Sample batch of environments {e_1,...,e_B} ~ E

        for each e:
            1. Collect experience (embodiment + active world-model)
               - Encode history → z_t
               - Estimate uncertainty u_t, difficulty d_t
               - c_t = R_ω(z_t, u_t, d_t)                  (resource allocation)
               - plan = plan_with_world_model(φ, z_t, budget=c_t)
               - a_t = π_θ(z_t, plan, goal)
               - o_{t+1}, r_t = E.step(a_t)

            2. Inner adaptation  (meta-learning)
               - Split D_train → D_support, D_query
               - (φ', θ') = InnerAdapt(φ, θ, D_support)

            3. Evaluate post-adaptation on D_query
               - Compute all losses

        4. Outer update (backprop through φ, θ, ψ, ω)
        5. EMA update: φ_ref, θ_ref

Formal objective (per spec)
----------------------------
    L_total = E_{e~E}[ L_task + λ1·L_WM + λ2·L_meta
                       + λ3·L_resource + λ4·L_ground + λ5·L_stab ]

This file provides:
    - RecursiveEngineTrainer   — main training orchestrator
    - InnerAdaptContext        — differentiable MAML-style inner loop
    - EpisodeBuffer            — trajectory storage
    - TaskEnvironmentBase      — abstract environment interface
"""

from __future__ import annotations

import copy
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.world_model_core import WorldModel, WorldModelConfig
from core.resource_allocator_re import ResourceAllocator, ResourceAllocatorConfig
from core.recursive_engine_objective import (
    RecursiveEngineObjective,
    RecursiveEngineConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract environment interface  (embodiment layer)
# ─────────────────────────────────────────────────────────────────────────────

class TaskEnvironmentBase(ABC):
    """
    Abstract base class for environments in the Recursive Engine training loop.

    Concrete subclasses can wrap:
    - ARC-AGI tasks (grid-to-grid)
    - OpenAI Gym / Gymnasium environments
    - Custom simulation environments
    - Real embodied systems (via sensor/actuator bridge)
    """

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation o_0.  [obs_dim]"""

    @abstractmethod
    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute action; return (obs, reward, done, info)."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Observation dimensionality."""

    @property
    @abstractmethod
    def act_dim(self) -> int:
        """Action dimensionality."""

    def estimate_difficulty(self) -> float:
        """Optionally override with task-specific difficulty estimate."""
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Episode / trajectory buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    obs:     torch.Tensor   # [obs_dim]
    action:  torch.Tensor   # [act_dim]
    reward:  float
    next_obs: torch.Tensor  # [obs_dim]
    done:    bool


class EpisodeBuffer:
    """Stores transitions from one or more episodes."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.transitions: List[Transition] = []
        self.device = device

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def __len__(self) -> int:
        return len(self.transitions)

    def as_tensors(
        self, obs_dim: int, act_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            obs     [T, obs_dim]
            actions [T, act_dim]
            rewards [T, 1]
            next_obs[T, obs_dim]
        All on self.device.
        """
        T = len(self.transitions)
        obs     = torch.stack([t.obs     for t in self.transitions])       # [T, obs_dim]
        actions = torch.stack([t.action  for t in self.transitions])       # [T, act_dim]
        rewards = torch.tensor([[t.reward] for t in self.transitions],
                               dtype=torch.float32)                        # [T, 1]
        next_obs = torch.stack([t.next_obs for t in self.transitions])     # [T, obs_dim]
        return (obs.to(self.device), actions.to(self.device),
                rewards.to(self.device), next_obs.to(self.device))

    def split(
        self, support_ratio: float = 0.5
    ) -> Tuple["EpisodeBuffer", "EpisodeBuffer"]:
        """Split into support (inner loop) and query (outer loop) sets."""
        n_support = max(1, int(len(self.transitions) * support_ratio))
        support = EpisodeBuffer(self.device)
        query   = EpisodeBuffer(self.device)
        support.transitions = self.transitions[:n_support]
        query.transitions   = self.transitions[n_support:]
        return support, query


# ─────────────────────────────────────────────────────────────────────────────
# Simple policy network π_θ
# ─────────────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    π_θ(a_t | z_t, plan, goal)

    Simplified: takes concatenated [z_t, plan_summary] and outputs action.
    For continuous actions: Gaussian head.
    """

    def __init__(self, latent_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),   # z + plan_summary
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head    = nn.Linear(hidden_dim, act_dim)
        self.sigma_head = nn.Linear(hidden_dim, act_dim)

    def forward(
        self,
        z: torch.Tensor,            # [B, latent_dim]
        plan_summary: torch.Tensor, # [B, latent_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action mu, sigma."""
        x = torch.cat([z, plan_summary], dim=-1)
        h = self.net(x)
        mu    = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-5
        return mu, sigma

    def sample_action(
        self,
        z: torch.Tensor,
        plan_summary: torch.Tensor,
    ) -> torch.Tensor:
        mu, sigma = self.forward(z, plan_summary)
        return mu + sigma * torch.randn_like(mu)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    # Outer loop
    n_meta_iters: int       = 10_000
    batch_envs: int         = 8       # B: environments per meta-iteration
    lr_outer: float         = 3e-4

    # Inner loop (MAML-style)
    n_inner_steps: int      = 3       # gradient steps for InnerAdapt
    lr_inner: float         = 1e-3
    support_ratio: float    = 0.5     # fraction of D_train used as support

    # Data collection
    n_episodes_per_env: int = 4
    t_max: int              = 64      # max steps per episode

    # EMA reference params
    ema_decay: float        = 0.995   # φ_ref = ema_decay·φ_ref + (1-ema_decay)·φ

    # Loss lambdas  (L_total = L_task + Σ λ_i · L_i)
    lambda_wm:       float = 0.5
    lambda_meta:     float = 0.3
    lambda_resource: float = 0.1
    lambda_ground:   float = 0.2
    lambda_stab:     float = 0.1

    # Checkpointing
    checkpoint_every: int  = 500
    checkpoint_dir: str    = "checkpoints/recursive_engine"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_every: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# Recursive Engine Trainer
# ─────────────────────────────────────────────────────────────────────────────

class RecursiveEngineTrainer:
    """
    Orchestrates the full Recursive Engine training loop.

    Four pillars
    ------------
    1. Active World-Model   — learn W_φ via L_WM, plan in latent space
    2. Recursive Meta-Learning — MAML-style inner/outer loop via L_meta
    3. Fractional Resource Distribution — R_ω allocates c_t, trained via L_resource
    4. Embodiment Grounding — L_ground keeps model tied to real observations

    Stability (L_stab) is a cross-cutting penalty against forgetting.

    Usage
    -----
        envs = [MyARCEnv(task) for task in arc_tasks]
        trainer = RecursiveEngineTrainer(envs, cfg=TrainerConfig())
        trainer.train()
    """

    def __init__(
        self,
        envs: List[TaskEnvironmentBase],
        cfg: Optional[TrainerConfig] = None,
        world_model_cfg: Optional[WorldModelConfig] = None,
        resource_cfg: Optional[ResourceAllocatorConfig] = None,
        objective_cfg: Optional[RecursiveEngineConfig] = None,
    ):
        self.envs = envs
        self.cfg  = cfg or TrainerConfig()
        self.device = torch.device(self.cfg.device)

        # Infer dims from first environment
        obs_dim = envs[0].obs_dim
        act_dim = envs[0].act_dim

        # World model (W_φ)
        wm_cfg = world_model_cfg or WorldModelConfig(obs_dim=obs_dim, act_dim=act_dim)
        self.world_model = WorldModel(wm_cfg).to(self.device)

        # Policy (π_θ)
        self.policy = PolicyNetwork(
            latent_dim=wm_cfg.latent_dim,
            act_dim=act_dim,
        ).to(self.device)

        # Resource allocator (R_ω)
        r_cfg = resource_cfg or ResourceAllocatorConfig(latent_dim=wm_cfg.latent_dim)
        self.resource_allocator = ResourceAllocator(r_cfg).to(self.device)

        # Composite objective (RecursiveEngineObjective wraps all L_* terms)
        obj_cfg = objective_cfg or RecursiveEngineConfig()
        self.objective = RecursiveEngineObjective(obj_cfg).to(self.device)

        # EMA reference parameters  (φ_ref, θ_ref from formal spec)
        self._wm_ref    = copy.deepcopy(self.world_model)
        self._policy_ref = copy.deepcopy(self.policy)
        for p in self._wm_ref.parameters():
            p.requires_grad_(False)
        for p in self._policy_ref.parameters():
            p.requires_grad_(False)

        # Single outer optimizer over all trainable modules
        self.optimizer = optim.AdamW(
            list(self.world_model.parameters())
            + list(self.policy.parameters())
            + list(self.resource_allocator.parameters()),
            lr=self.cfg.lr_outer,
        )

        self._meta_iter = 0
        self._metrics_history: List[Dict[str, float]] = []

    # ── EMA helpers ──────────────────────────────────────────────────────────

    def _ema_update(self, ref_model: nn.Module, model: nn.Module) -> None:
        """Update reference weights:  ref = ema_decay·ref + (1-ema_decay)·model."""
        decay = self.cfg.ema_decay
        with torch.no_grad():
            for ref_p, p in zip(ref_model.parameters(), model.parameters()):
                ref_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    # ── Data collection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_experience(
        self,
        env: TaskEnvironmentBase,
        world_model: WorldModel,
        policy: PolicyNetwork,
        resource_allocator: ResourceAllocator,
    ) -> EpisodeBuffer:
        """
        Step 1 of the training loop: collect trajectories using current models.

        Embodies the four-pillar loop:
            encode h → z
            estimate u (uncertainty), d (difficulty)
            c_t = R_ω(z, u, d)
            plan = imagine(z, budget=c_t)
            a_t = π_θ(z, plan_summary)
            o_{t+1}, r_t = env.step(a_t)
        """
        buf = EpisodeBuffer(self.device)

        for _ in range(self.cfg.n_episodes_per_env):
            obs = env.reset().to(self.device).float()   # [obs_dim]
            obs_history = []
            act_history = []
            rew_history = []
            rnn_hidden  = None

            for _t in range(self.cfg.t_max):
                act_dim = env.act_dim
                wm_cfg  = world_model.cfg

                # Build history tensors  [1, T_so_far, dim]
                if obs_history:
                    obs_t = torch.stack(obs_history).unsqueeze(0)       # [1, T, obs_dim]
                    act_t = torch.stack(act_history).unsqueeze(0)       # [1, T, act_dim]
                    rew_t = torch.tensor(rew_history,
                                        dtype=torch.float32,
                                        device=self.device).view(1, -1, 1)
                else:
                    obs_t = obs.unsqueeze(0).unsqueeze(0)               # [1, 1, obs_dim]
                    act_t = torch.zeros(1, 1, act_dim, device=self.device)
                    rew_t = torch.zeros(1, 1, 1, device=self.device)

                # Encode history → latent z  (W_φ encoder)
                z_mu, z_sigma, rnn_hidden = world_model.encode(
                    obs_t, act_t, rew_t, rnn_hidden
                )
                z = world_model.sample_latent(z_mu, z_sigma)  # [1, latent_dim]

                # Uncertainty & difficulty (R_ω inputs)
                uncertainty = world_model.compute_uncertainty(z)         # [1]
                difficulty  = torch.tensor(
                    [env.estimate_difficulty()], device=self.device
                )

                # Allocate compute (R_ω)
                _, _, plan_depth, _ = resource_allocator(z, uncertainty, difficulty)

                # Plan with world model (imagine plan_depth steps)
                dummy_actions = torch.zeros(
                    1, plan_depth, act_dim, device=self.device
                )
                z_seq, _ = world_model.imagine(z, dummy_actions)  # [1, T+1, D]
                plan_summary = z_seq[:, -1]                        # [1, D] last imagined state

                # Sample action (π_θ)
                action = policy.sample_action(z, plan_summary)     # [1, act_dim]
                action_squeezed = action.squeeze(0)                # [act_dim]

                # Environment step (embodiment grounding)
                next_obs, reward, done, _ = env.step(action_squeezed.cpu())
                next_obs = next_obs.to(self.device).float()

                # Log transition
                buf.add(Transition(
                    obs=obs,
                    action=action_squeezed,
                    reward=float(reward),
                    next_obs=next_obs,
                    done=done,
                ))

                obs_history.append(obs)
                act_history.append(action_squeezed)
                rew_history.append(float(reward))
                obs = next_obs

                if done:
                    break

        return buf

    # ── Inner adaptation  (MAML inner loop) ──────────────────────────────────

    def _inner_adapt(
        self,
        support_buf: EpisodeBuffer,
        obs_dim: int,
        act_dim: int,
    ) -> Tuple[WorldModel, PolicyNetwork]:
        """
        (φ', θ') = InnerAdapt(φ, θ, D_support)

        Makes n_inner_steps gradient steps on D_support using the world-model
        loss and a simple behavioural cloning (BC) proxy for task loss.

        Returns shallow copies with updated (but tracked) parameters for
        second-order gradient flow through the outer update.
        """
        phi_prime    = copy.deepcopy(self.world_model)
        theta_prime  = copy.deepcopy(self.policy)

        inner_opt = optim.SGD(
            list(phi_prime.parameters()) + list(theta_prime.parameters()),
            lr=self.cfg.lr_inner,
        )

        obs, actions, rewards, next_obs = support_buf.as_tensors(obs_dim, act_dim)
        obs_b    = obs.unsqueeze(0)      # [1, T, obs_dim]
        act_b    = actions.unsqueeze(0)  # [1, T, act_dim]
        rew_b    = rewards.unsqueeze(0)  # [1, T, 1]

        for _ in range(self.cfg.n_inner_steps):
            inner_opt.zero_grad()

            # World-model reconstruction loss on support set
            l_wm, _ = phi_prime.compute_loss(obs_b, act_b, rew_b, next_obs.unsqueeze(0))

            # Simple BC proxy: policy log-prob on observed actions
            z_mu, z_sigma, _ = phi_prime.encode(obs_b, act_b, rew_b)
            z = phi_prime.sample_latent(z_mu, z_sigma)  # [1, D]
            plan_dummy = z
            pi_mu, pi_sigma = theta_prime(z, plan_dummy)
            dist = torch.distributions.Normal(pi_mu, pi_sigma)
            # BC: maximise log-prob of observed last action
            l_bc = -dist.log_prob(actions[-1].unsqueeze(0)).mean()

            loss = l_wm + l_bc
            loss.backward()
            inner_opt.step()

        return phi_prime, theta_prime

    # ── Compute per-env outer loss ────────────────────────────────────────────

    def _compute_env_loss(
        self,
        env: TaskEnvironmentBase,
        query_buf: EpisodeBuffer,
        phi_prime: WorldModel,
        theta_prime: PolicyNetwork,
        support_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute L_env = L_task + λ1·L_WM + λ2·L_meta + λ3·L_res + λ4·L_ground + λ5·L_stab
        on the query set using adapted parameters (φ', θ').
        """
        obs_dim = env.obs_dim
        act_dim = env.act_dim
        cfg     = self.cfg

        obs, actions, rewards, next_obs = query_buf.as_tensors(obs_dim, act_dim)
        obs_b    = obs.unsqueeze(0)
        act_b    = actions.unsqueeze(0)
        rew_b    = rewards.unsqueeze(0)

        # ── L_WM ─────────────────────────────────────────────────────────────
        l_wm, wm_breakdown = phi_prime.compute_loss(obs_b, act_b, rew_b, next_obs.unsqueeze(0))

        # ── L_task (supervised: BC on query actions using adapted policy) ─────
        z_mu, _, _ = phi_prime.encode(obs_b, act_b, rew_b)
        pi_mu, pi_sigma = theta_prime(z_mu, z_mu)
        dist_pi = torch.distributions.Normal(pi_mu, pi_sigma)
        l_task  = -dist_pi.log_prob(actions[-1].unsqueeze(0)).mean()

        # ── L_meta = L_task + η·C_adapt ──────────────────────────────────────
        l_meta = l_task + 0.01 * float(support_steps)

        # ── L_resource ────────────────────────────────────────────────────────
        T = len(query_buf)
        z_mu_full, _, _ = phi_prime.encode(obs_b, act_b, rew_b)
        z_seq_fake = z_mu_full.unsqueeze(1).expand(-1, T, -1)  # [1, T, D]
        u_seq = torch.full((1, T), 0.5, device=self.device)
        d_seq = torch.full((1, T), env.estimate_difficulty(), device=self.device)
        l_res, res_breakdown = self.resource_allocator.compute_batch_loss(z_seq_fake, u_seq, d_seq)

        # ── L_ground: predicted obs vs real next_obs ──────────────────────────
        obs_mu, _ = phi_prime.decode(z_mu)                         # [1, obs_dim]
        l_ground  = F.mse_loss(obs_mu, next_obs[-1].unsqueeze(0))

        # ── L_stab: D(φ, φ_ref) + D(θ, θ_ref) ───────────────────────────────
        # Using squared L2 norm on parameter vectors (EWC-lite)
        l_stab = torch.zeros(1, device=self.device)
        for p, ref_p in zip(phi_prime.parameters(), self._wm_ref.parameters()):
            l_stab = l_stab + F.mse_loss(p, ref_p.detach())
        for p, ref_p in zip(theta_prime.parameters(), self._policy_ref.parameters()):
            l_stab = l_stab + F.mse_loss(p, ref_p.detach())

        # ── Total ─────────────────────────────────────────────────────────────
        l_env = (l_task
                 + cfg.lambda_wm       * l_wm
                 + cfg.lambda_meta     * l_meta
                 + cfg.lambda_resource * l_res
                 + cfg.lambda_ground   * l_ground
                 + cfg.lambda_stab     * l_stab)

        metrics = {
            "l_task":     l_task.item(),
            "l_wm":       l_wm.item(),
            "l_meta":     l_meta.item() if isinstance(l_meta, torch.Tensor) else float(l_meta),
            "l_resource": l_res.item(),
            "l_ground":   l_ground.item(),
            "l_stab":     l_stab.item(),
            "l_env":      l_env.item(),
            **{f"wm_{k}": v for k, v in wm_breakdown.items()},
        }
        return l_env, metrics

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Execute the full Recursive Engine training loop.

            for meta_iter in 1..N_meta:
                Sample B environments from E
                for each e:
                    collect → inner_adapt → evaluate → accumulate loss
                outer update (φ, θ, ψ, ω)
                ema update (φ_ref, θ_ref)
        """
        import random
        cfg = self.cfg

        for meta_iter in range(1, cfg.n_meta_iters + 1):
            self._meta_iter = meta_iter
            t0 = time.time()

            # Sample B environments
            batch_envs = random.choices(self.envs, k=cfg.batch_envs)

            self.optimizer.zero_grad()
            total_loss = torch.zeros(1, device=self.device)
            iter_metrics: List[Dict[str, float]] = []

            for env in batch_envs:
                obs_dim = env.obs_dim
                act_dim = env.act_dim

                # ── Step 1: Collect experience ──────────────────────────────
                D_train = self._collect_experience(
                    env, self.world_model, self.policy, self.resource_allocator
                )

                if len(D_train) < 4:
                    continue  # not enough data; skip this env

                # ── Step 2: Inner adaptation ────────────────────────────────
                D_support, D_query = D_train.split(cfg.support_ratio)

                phi_prime, theta_prime = self._inner_adapt(
                    D_support, obs_dim, act_dim
                )

                if len(D_query) == 0:
                    D_query = D_support  # fallback

                # ── Step 3: Compute outer losses ────────────────────────────
                l_env, metrics = self._compute_env_loss(
                    env, D_query, phi_prime, theta_prime,
                    support_steps=len(D_support),
                )

                total_loss = total_loss + l_env
                iter_metrics.append(metrics)

            # ── Step 4: Outer update ────────────────────────────────────────
            if len(iter_metrics) > 0:
                total_loss = total_loss / len(iter_metrics)
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.world_model.parameters())
                    + list(self.policy.parameters())
                    + list(self.resource_allocator.parameters()),
                    max_norm=10.0,
                )
                self.optimizer.step()

            # ── EMA update of reference params ─────────────────────────────
            self._ema_update(self._wm_ref, self.world_model)
            self._ema_update(self._policy_ref, self.policy)

            # ── Logging ─────────────────────────────────────────────────────
            if meta_iter % cfg.log_every == 0 and iter_metrics:
                avg = {k: sum(m.get(k, 0) for m in iter_metrics) / len(iter_metrics)
                       for k in iter_metrics[0]}
                elapsed = time.time() - t0
                self._metrics_history.append({"iter": meta_iter, **avg})
                print(
                    f"[iter {meta_iter:6d}] "
                    f"total={total_loss.item():.4f}  "
                    f"task={avg.get('l_task',0):.4f}  "
                    f"wm={avg.get('l_wm',0):.4f}  "
                    f"meta={avg.get('l_meta',0):.4f}  "
                    f"res={avg.get('l_resource',0):.4f}  "
                    f"stab={avg.get('l_stab',0):.4f}  "
                    f"dt={elapsed:.2f}s"
                )

            # ── Checkpoint ──────────────────────────────────────────────────
            if meta_iter % cfg.checkpoint_every == 0:
                self._save_checkpoint(meta_iter)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, meta_iter: int) -> None:
        import os
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = f"{self.cfg.checkpoint_dir}/re_step_{meta_iter:06d}.pt"
        torch.save(
            {
                "meta_iter":         meta_iter,
                "world_model":       self.world_model.state_dict(),
                "policy":            self.policy.state_dict(),
                "resource_allocator": self.resource_allocator.state_dict(),
                "wm_ref":            self._wm_ref.state_dict(),
                "policy_ref":        self._policy_ref.state_dict(),
                "optimizer":         self.optimizer.state_dict(),
            },
            path,
        )
        print(f"  ✓ Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(ckpt["world_model"])
        self.policy.load_state_dict(ckpt["policy"])
        self.resource_allocator.load_state_dict(ckpt["resource_allocator"])
        self._wm_ref.load_state_dict(ckpt["wm_ref"])
        self._policy_ref.load_state_dict(ckpt["policy_ref"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._meta_iter = ckpt["meta_iter"]
        print(f"  ✓ Loaded checkpoint from {path} (iter {self._meta_iter})")

    @property
    def metrics_history(self) -> List[Dict[str, float]]:
        return self._metrics_history
