"""
Compound Loop Controller

Adaptive looped reasoning inspired by Ouro (Scaling Latent Reasoning via
Looped Language Models). Wraps the limb processing pipeline in a learnable
loop with:
  - Per-step exit gate (sigmoid + survival probability math)
  - Entropy regularization to prevent collapse to single loop
  - Weighted loss across all loop depths during training
  - Adaptive early exit during inference

"Compound" because it loops the entire compound braid + limb pipeline,
giving the model variable compute per input depending on difficulty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .recurrent_depth_transformer import RecurrentDepthTransformer
from .adaptive_computation_controller import (
    AdaptiveComputationController,
    AdaptiveComputationConfig,
)


@dataclass
class CompoundLoopConfig:
    max_loops: int = 4
    exit_threshold: float = 0.5
    entropy_beta: float = 0.1  # KL regularization strength
    prior: str = 'uniform'     # 'uniform' >> 'geometric' (paper finding)
    warmup_loops: int = 0      # force minimum loops before exit gate activates
    # RIRM: Reflection Inhibition Reward (Yuan 3.0 Ultra inspired)
    conciseness_reward: float = 0.05   # reward for exiting early
    max_cheap_loops: int = 2           # loops beyond this get penalized
    # Recurrent Depth Transformer integration
    use_recurrent_depth: bool = True   # enable RDT for adaptive depth
    rdt_hidden_dim: int = 128          # RDT hidden dimension (typically hidden_dim // 2)
    rdt_num_heads: int = 4             # RDT attention heads
    rdt_num_layers: int = 2            # RDT transformer layers
    rdt_depth_loss_weight: float = 0.01  # weight for depth regularization loss
    # Adaptive Computation Time (ACT) integration
    use_adaptive_computation: bool = True  # enable ACT for adaptive budget
    act_base_budget: float = 1.0           # base ponder cost per loop
    act_budget_from_uncertainty: bool = True  # scale budget by RDT uncertainty
    act_uncertainty_budget_scale: float = 2.0  # max budget multiplier
    act_cost_loss_weight: float = 0.01      # weight for ACT cost loss
    act_cost_target: float = 0.5           # target average ponder cost
    act_intensity_scaling: bool = True      # scale routing by remaining budget
    act_enable_budget_pressure: bool = True # pressure halting based on budget


class ExitGate(nn.Module):
    """Learned exit gate: Linear → Sigmoid → P(exit at this step)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-batch exit probability from pooled representation."""
        # x: [batch, seq, hidden] → pool → [batch, 1]
        pooled = x.mean(dim=1)
        return self.gate(pooled).squeeze(-1)  # [batch]


class CompoundLoopController(nn.Module):
    """
    Manages adaptive looping over a processing block.

    During training: runs all loops, computes weighted loss using exit
    distribution, adds entropy regularization.

    During inference: exits early when CDF > threshold.
    
    Now integrated with RecurrentDepthTransformer for intelligent depth control.
    """

    def __init__(self, hidden_dim: int, config: Optional[CompoundLoopConfig] = None):
        super().__init__()
        self.config = config or CompoundLoopConfig()
        self.hidden_dim = hidden_dim
        self.max_loops = self.config.max_loops
        self.exit_threshold = self.config.exit_threshold
        self.entropy_beta = self.config.entropy_beta
        self.warmup_loops = self.config.warmup_loops

        # Exit gates for loops 0..max_loops-2 (final step force-exits with remaining mass)
        self.exit_gates = nn.ModuleList([
            ExitGate(hidden_dim) for _ in range(max(1, self.max_loops - 1))
        ])

        # Lightweight loop-step embedding so the block knows which iteration it's on
        self.loop_embed = nn.Embedding(self.max_loops, hidden_dim)

        # Layer norm to stabilize looped representations
        self.loop_norm = nn.LayerNorm(hidden_dim)

        # Residual scaling — start small so early training is stable
        self.loop_alpha = nn.Parameter(torch.tensor(0.1))
        
        # === NEW: Recurrent Depth Transformer for adaptive depth control ===
        self.use_recurrent_depth = self.config.use_recurrent_depth
        if self.use_recurrent_depth:
            self.rdt = RecurrentDepthTransformer(
                hidden_dim=hidden_dim,
                num_heads=self.config.rdt_num_heads,
                num_layers=self.config.rdt_num_layers,
                ffn_dim=hidden_dim * 2,
                dropout=0.1,
                num_limbs=14,  # 11 cognitive limbs + 3 modalities
                max_depth=self.max_loops,
            )
        else:
            self.rdt = None
        
        # === NEW: Adaptive Computation Time (ACT) for budget-aware routing ===
        self.use_adaptive_computation = self.config.use_adaptive_computation
        if self.use_adaptive_computation:
            act_config = AdaptiveComputationConfig(
                base_budget=self.config.act_base_budget,
                budget_from_uncertainty=self.config.act_budget_from_uncertainty,
                uncertainty_budget_scale=self.config.act_uncertainty_budget_scale,
                halting_hidden_dim=self.config.rdt_hidden_dim,
                halting_threshold=0.99,
                cost_loss_weight=self.config.act_cost_loss_weight,
                cost_target=self.config.act_cost_target,
                intensity_scaling=self.config.act_intensity_scaling,
                enable_budget_pressure=self.config.act_enable_budget_pressure,
            )
            self.act_controller = AdaptiveComputationController(
                hidden_dim=hidden_dim,
                config=act_config,
                max_loops=self.max_loops,
            )
        else:
            self.act_controller = None

        # Track stats
        self._last_loop_count = 0
        self._last_exit_dist: List[float] = []
        self._last_routing_gates = None
        self._last_uncertainties = []
        self._last_budgets = None
        self._last_ponder_costs = None
        self._last_routing_intensities = None

    def forward(
        self,
        x: torch.Tensor,
        process_fn,
        process_kwargs: Optional[Dict] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rdt_uncertainties: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run process_fn in an adaptive loop with RDT + ACT integration.

        Args:
            x: Input tensor [batch, seq, hidden]
            process_fn: Callable(x, loop_idx, **kwargs) → x_processed
            process_kwargs: Extra kwargs passed to process_fn
            attention_mask: Optional attention mask for RDT
            rdt_uncertainties: [batch] uncertainty from RDT for budget allocation

        Returns:
            Dict with full ACT + RDT information
        """
        process_kwargs = process_kwargs or {}
        B = x.size(0)
        device = x.device

        exit_probs = []      # sigmoid outputs per step
        loop_outputs = []    # processed representations per step
        survival = torch.ones(B, device=device)
        cdf = torch.zeros(B, device=device)
        
        # RDT and ACT tracking
        rdt_uncertainties_list = []
        rdt_routing_history = []
        routing_intensities_history = []
        
        # Initialize ACT if enabled
        if self.use_adaptive_computation and self.act_controller is not None:
            self.act_controller.start_computation(B)
            # Allocate budget based on RDT uncertainties
            budgets = self.act_controller.allocate_budget(B, rdt_uncertainties, device)
            self._last_budgets = budgets
        else:
            budgets = None

        base_x = x  # preserve for residual

        for loop_idx in range(self.max_loops):
            # Add loop position embedding
            loop_emb = self.loop_embed(
                torch.tensor(loop_idx, device=device)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
            x_input = x + loop_emb

            # Process this loop iteration
            x_processed = process_fn(x_input, loop_idx, **process_kwargs)

            # Residual connection with learnable scaling
            x = self.loop_norm(base_x + self.loop_alpha * (x_processed - base_x))

            loop_outputs.append(x)
            
            # === NEW: RDT-guided depth decision ===
            rdt_info = None
            if self.use_recurrent_depth and self.rdt is not None:
                rdt_info = self.rdt(x, depth=loop_idx, attention_mask=attention_mask)
                rdt_uncertainties_list.append(rdt_info['uncertainty'])
                rdt_routing_history.append(rdt_info['routing_gates'])
                
                # Store routing gates for use by CompoundBraid
                self._last_routing_gates = rdt_info['routing_gates']
            
            # === NEW: ACT halting probability and budget-aware routing ===
            halting_prob = None
            routing_intensity = None
            if self.use_adaptive_computation and self.act_controller is not None:
                halting_prob, cumulative_halt, alive_prob = self.act_controller.should_halt(
                    x, loop_idx, budgets, training=self.training
                )
                routing_intensity = self.act_controller.get_routing_intensity(loop_idx, budgets, self.max_loops)
                routing_intensities_history.append(routing_intensity)
                self._last_routing_intensities = routing_intensity

            # Exit gate (last step has no gate — it gets remaining mass)
            is_last = (loop_idx == self.max_loops - 1)
            if not is_last:
                # Determine exit probability: blend RDT + ACT + standard gate
                if self.use_recurrent_depth and rdt_info is not None:
                    p_exit_rdt = rdt_info['exit_prob']  # [batch]
                else:
                    p_exit_rdt = None
                
                if halting_prob is not None:
                    # Use ACT halting as primary decision
                    p_exit = halting_prob
                    if p_exit_rdt is not None:
                        # Blend with RDT: 60% ACT + 40% RDT
                        p_exit = 0.6 * p_exit + 0.4 * p_exit_rdt
                elif p_exit_rdt is not None:
                    # Only RDT available
                    p_exit_gate = self.exit_gates[loop_idx](x)  # [batch]
                    p_exit = 0.7 * p_exit_rdt + 0.3 * p_exit_gate
                else:
                    # Standard exit gate only
                    p_exit = self.exit_gates[loop_idx](x)  # [batch]
                
                p_exit = p_exit.clamp(1e-6, 1 - 1e-6)
                exit_probs.append(p_exit)

                unconditional = survival * p_exit
                cdf = cdf + unconditional
                survival = survival * (1 - p_exit)

                # Inference early exit (after warmup loops)
                if not self.training and loop_idx >= self.warmup_loops:
                    if cdf.mean() > self.exit_threshold:
                        self._last_loop_count = loop_idx + 1
                        self._last_uncertainties = [u.mean().item() for u in rdt_uncertainties_list]
                        result = {
                            'output': x,
                            'loop_count': loop_idx + 1,
                            'exit_distribution': [p.mean().item() for p in exit_probs],
                            'entropy_loss': torch.tensor(0.0, device=device),
                            'loop_outputs': None,
                            'rdt_routing_gates': self._last_routing_gates,
                            'rdt_uncertainties': self._last_uncertainties,
                        }
                        # Add ACT info if available
                        if self.use_adaptive_computation and self.act_controller is not None:
                            result['act_budgets'] = budgets.tolist() if budgets is not None else None
                            result['act_ponder_costs'] = self.act_controller.get_ponder_cost().tolist()
                            result['act_routing_intensities'] = [r.mean().item() for r in routing_intensities_history] if routing_intensities_history else None
                        return result

        # --- Training: compute proper exit distribution & weighted output ---
        # P(t) = S(t) * p_exit(t) for gated steps, remaining mass for final step
        # exit_probs has max_loops-1 entries (one per gate)
        P = []
        s = torch.ones(B, device=device)
        for p in exit_probs:  # all gated steps
            P.append(s * p)
            s = s * (1 - p)
        P.append(s)  # remaining mass → final step (forced exit, no gate)
        # len(P) == max_loops == len(loop_outputs)

        # Weighted combination of all loop outputs
        # P[i]: [batch], loop_outputs[i]: [batch, seq, hidden]
        weighted_output = torch.zeros_like(loop_outputs[0])
        for p_t, out_t in zip(P, loop_outputs):
            weighted_output = weighted_output + p_t.unsqueeze(-1).unsqueeze(-1) * out_t

        # Entropy regularization (KL vs uniform prior)
        entropy_loss = self._entropy_regularization(P, device)

        # RIRM: penalize unnecessary loops (Yuan 3.0 Ultra inspired)
        # If high exit probability early, reward conciseness
        conciseness_loss = self._conciseness_penalty(P, device)
        entropy_loss = entropy_loss + conciseness_loss
        
        # === NEW: RDT depth regularization loss ===
        rdt_depth_loss = torch.tensor(0.0, device=device)
        if self.use_recurrent_depth and rdt_uncertainties_list:
            # Stack uncertainties: [max_loops]
            uncertainties = torch.stack(rdt_uncertainties_list)  # [max_loops]
            # Regularize: prefer lower uncertainty (more committed decisions)
            rdt_depth_loss = self.config.rdt_depth_loss_weight * uncertainties.mean()
            entropy_loss = entropy_loss + rdt_depth_loss
        
        # === NEW: ACT cost regularization loss ===
        act_cost_loss = torch.tensor(0.0, device=device)
        if self.use_adaptive_computation and self.act_controller is not None:
            act_cost_loss = self.act_controller.get_cost_loss()
            entropy_loss = entropy_loss + act_cost_loss

        # Track stats
        self._last_loop_count = self.max_loops
        self._last_exit_dist = [p.mean().item() for p in P]
        self._last_uncertainties = [u.mean().item() for u in rdt_uncertainties_list] if rdt_uncertainties_list else []
        if rdt_routing_history:
            self._last_routing_gates = rdt_routing_history[-1]  # Use last routing
        if routing_intensities_history:
            self._last_routing_intensities = torch.stack(routing_intensities_history).mean(dim=0)

        result = {
            'output': weighted_output,
            'loop_count': self.max_loops,
            'exit_distribution': self._last_exit_dist,
            'entropy_loss': entropy_loss,
            'loop_outputs': loop_outputs,
            'rdt_routing_gates': self._last_routing_gates,
            'rdt_uncertainties': self._last_uncertainties,
            'rdt_depth_loss': rdt_depth_loss,
        }
        
        # Add ACT information if enabled
        if self.use_adaptive_computation and self.act_controller is not None:
            result.update({
                'act_budgets': budgets.tolist() if budgets is not None else None,
                'act_ponder_costs': self.act_controller.get_ponder_cost().tolist(),
                'act_cost_loss': act_cost_loss,
                'act_routing_intensities': [r.mean().item() for r in routing_intensities_history] if routing_intensities_history else None,
            })
        
        return result

    def _entropy_regularization(
        self, P: List[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """KL(exit_dist || uniform_prior) — penalizes collapsed distributions."""
        K = len(P)
        if K <= 1:
            return torch.tensor(0.0, device=device)

        # Stack: [batch, K]
        exit_dist = torch.stack(P, dim=-1)
        exit_dist = exit_dist.clamp(min=1e-8)  # prevent log(0)

        # Uniform prior
        uniform = torch.ones(K, device=device) / K

        # KL divergence: sum p * log(p/q)
        kl = (exit_dist * (exit_dist.log() - uniform.log())).sum(dim=-1)

        return self.entropy_beta * kl.mean()

    def _conciseness_penalty(
        self, P: List[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """
        RIRM: Reflection Inhibition Reward Mechanism.
        Penalizes probability mass on loops beyond max_cheap_loops.
        Encourages the model to exit early when possible.
        """
        if self.config.conciseness_reward <= 0 or len(P) <= self.config.max_cheap_loops:
            return torch.tensor(0.0, device=device)

        # Sum probability mass on "expensive" late loops
        late_mass = torch.zeros(P[0].shape[0], device=device)
        for i in range(self.config.max_cheap_loops, len(P)):
            late_mass = late_mass + P[i]

        # Penalty = reward_weight * mean late-loop probability mass
        return self.config.conciseness_reward * late_mass.mean()

    def get_stats(self) -> Dict:
        stats = {
            'last_loop_count': self._last_loop_count,
            'last_exit_distribution': self._last_exit_dist,
            'loop_alpha': self.loop_alpha.item()
        }
        
        # Add RDT stats if available
        if self.use_recurrent_depth:
            stats.update({
                'rdt_uncertainties': self._last_uncertainties,
                'rdt_routing_gates': self._last_routing_gates.mean(dim=0).tolist() if self._last_routing_gates is not None else None,
            })
        
        # Add ACT stats if available
        if self.use_adaptive_computation and self.act_controller is not None:
            act_stats = self.act_controller.get_stats()
            stats.update({
                'act_budgets': act_stats.get('budgets'),
                'act_ponder_costs': act_stats.get('ponder_costs'),
                'act_routing_intensities': act_stats.get('routing_intensities'),
                'act_num_steps': act_stats.get('num_steps'),
                'act_halt_distribution': act_stats.get('halt_distribution'),
            })
        
        return stats
