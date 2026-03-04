"""
Compound MoE Integration — bridges the MoE architecture with
the compound learning engine for adaptive expert routing and
cross-expert knowledge transfer.

Each MoE expert is treated as a "model" in the compound learning
framework, enabling:
  - Learned expert affinity patterns
  - Cross-expert knowledge transfer tracking
  - Adaptive routing bias based on accumulated patterns
  - Expert specialization monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import math
import time
import json


@dataclass
class ExpertProfile:
    """Tracks compound learning statistics for a single expert."""
    expert_id: int
    total_tokens_routed: int = 0
    cumulative_reward: float = 0.0
    specialization_vector: Optional[List[float]] = None
    avg_output_norm: float = 0.0
    cross_transfer_scores: Dict[int, float] = field(default_factory=dict)
    pattern_hits: int = 0


@dataclass
class CompoundRoutingState:
    """Persistent state for compound-aware routing decisions."""
    expert_profiles: Dict[int, ExpertProfile] = field(default_factory=dict)
    affinity_matrix: Dict[str, float] = field(default_factory=dict)
    total_routing_steps: int = 0
    pattern_cache: Dict[str, List[int]] = field(default_factory=dict)


class ExpertCompoundTracker(nn.Module):
    """
    Tracks expert utilization and performance for compound learning feedback.
    Maintains running statistics that feed into adaptive routing.
    """

    def __init__(self, num_experts: int, hidden_dim: int, ema_decay: float = 0.99):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        # Running statistics (not parameters, but persistent state)
        self.register_buffer(
            "expert_load_ema",
            torch.ones(num_experts) / num_experts,
        )
        self.register_buffer(
            "expert_output_norm_ema",
            torch.ones(num_experts),
        )
        self.register_buffer(
            "expert_pair_coactivation",
            torch.zeros(num_experts, num_experts),
        )
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_outputs: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        """Update running statistics from a routing step."""
        self.total_steps += 1
        N = expert_indices.shape[0]

        # Update load EMA
        load = torch.zeros(self.num_experts, device=expert_indices.device)
        for k in range(expert_indices.shape[1]):
            load.scatter_add_(
                0,
                expert_indices[:, k],
                torch.ones(N, device=expert_indices.device),
            )
        load = load / max(N, 1)
        self.expert_load_ema.mul_(self.ema_decay).add_(
            load, alpha=1 - self.ema_decay
        )

        # Update co-activation counts (which experts are selected together)
        for k1 in range(expert_indices.shape[1]):
            for k2 in range(k1 + 1, expert_indices.shape[1]):
                pairs = torch.stack(
                    [expert_indices[:, k1], expert_indices[:, k2]], dim=1
                )
                for i in range(pairs.shape[0]):
                    e1, e2 = pairs[i, 0].item(), pairs[i, 1].item()
                    self.expert_pair_coactivation[e1, e2] += 1
                    self.expert_pair_coactivation[e2, e1] += 1

    def get_load_balance_score(self) -> float:
        """Returns 0-1 score; 1.0 = perfectly balanced."""
        ideal = 1.0 / self.num_experts
        deviation = (self.expert_load_ema - ideal).abs().mean().item()
        return max(0.0, 1.0 - deviation * self.num_experts)

    def get_expert_affinity_matrix(self) -> torch.Tensor:
        """Normalized co-activation matrix showing expert affinity."""
        total = self.expert_pair_coactivation.sum()
        if total > 0:
            return self.expert_pair_coactivation / total
        return self.expert_pair_coactivation

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps.item(),
            "load_balance_score": self.get_load_balance_score(),
            "expert_loads": self.expert_load_ema.tolist(),
            "top_expert_pairs": self._top_pairs(5),
        }

    def _top_pairs(self, k: int) -> List[Tuple[int, int, float]]:
        mat = self.expert_pair_coactivation
        flat = mat.flatten()
        vals, idxs = flat.topk(min(k, flat.numel()))
        pairs = []
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            e1 = idx // self.num_experts
            e2 = idx % self.num_experts
            pairs.append((e1, e2, v))
        return pairs


class CompoundRoutingBias(nn.Module):
    """
    Learnable routing bias that compounds over training.
    Adds a small adaptive bias to the MoE router logits based on
    accumulated expert performance signals.
    """

    def __init__(self, num_experts: int, hidden_dim: int, bias_scale: float = 0.1):
        super().__init__()
        self.bias_scale = bias_scale
        # Compact projection: hidden -> num_experts bias
        self.bias_proj = nn.Sequential(
            nn.Linear(hidden_dim, num_experts, bias=False),
            nn.Tanh(),
        )
        # Persistent compound bias accumulator (updated via compound learning)
        self.register_buffer(
            "compound_bias", torch.zeros(num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive routing bias.
        Args:
            x: [N, hidden_dim] token representations
        Returns:
            bias: [N, num_experts] additive bias for router logits
        """
        # Input-dependent bias
        input_bias = self.bias_proj(x)
        # Persistent compound bias (broadcast)
        return self.bias_scale * (input_bias + self.compound_bias.unsqueeze(0))

    @torch.no_grad()
    def update_compound_bias(self, expert_rewards: torch.Tensor, lr: float = 0.01):
        """
        Update the compound bias based on expert reward signals.
        Args:
            expert_rewards: [num_experts] reward signal per expert
            lr: compound learning rate
        """
        self.compound_bias.add_(expert_rewards * lr)
        # Normalize to prevent drift
        self.compound_bias.sub_(self.compound_bias.mean())


class CompoundMoELayer(nn.Module):
    """
    MoE layer enhanced with compounding integrations.

    Wraps the base MoELayer with:
      - ExpertCompoundTracker for utilization monitoring
      - CompoundRoutingBias for adaptive expert selection
      - Cross-expert knowledge transfer via shared representation projection
      - Pattern-based routing cache for recurring input types
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int = 64,
        top_k: int = 8,
        dropout: float = 0.1,
        jitter_noise: float = 0.01,
        compound_bias_scale: float = 0.1,
        enable_cross_transfer: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.enable_cross_transfer = enable_cross_transfer

        # Core MoE components (same as base)
        from core.moe import MoERouter, ExpertFFN

        self.router = MoERouter(hidden_dim, num_experts, top_k, jitter_noise)
        self.experts = nn.ModuleList(
            [ExpertFFN(hidden_dim, ffn_dim, dropout) for _ in range(num_experts)]
        )

        # Compound integration components
        self.compound_tracker = ExpertCompoundTracker(num_experts, hidden_dim)
        self.compound_bias = CompoundRoutingBias(num_experts, hidden_dim, compound_bias_scale)

        # Cross-expert knowledge transfer: shared low-rank projection
        if enable_cross_transfer:
            transfer_rank = min(64, hidden_dim // 4)
            self.transfer_down = nn.Linear(hidden_dim, transfer_rank, bias=False)
            self.transfer_up = nn.Linear(transfer_rank, hidden_dim, bias=False)
            self.transfer_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        braid_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with compound learning integration.

        Args:
            x: [batch, seq_len, hidden_dim]
            braid_signal: optional [num_experts] routing hint from CompoundBraid
        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_loss: scalar (load balance + compound regularization)
        """
        batch, seq_len, d = x.shape
        flat_x = x.reshape(-1, d)

        # Get base router logits, then add compound bias
        weights, indices, base_aux_loss = self.router(flat_x)

        # Apply compound routing bias (modulates logits before final selection)
        # Re-route with bias for better expert selection
        if self.training:
            bias = self.compound_bias(flat_x)
            # Add braid signal as additional routing hint
            if braid_signal is not None:
                bias = bias + 0.1 * braid_signal.unsqueeze(0)
            biased_logits = self.router.gate(flat_x) + bias
            top_k_logits, top_k_indices = torch.topk(
                biased_logits, self.top_k, dim=-1
            )
            weights = F.softmax(top_k_logits, dim=-1)
            indices = top_k_indices

        # Dispatch to experts
        output = torch.zeros_like(flat_x)
        expert_outputs_for_tracking: Dict[int, torch.Tensor] = {}

        for k in range(self.top_k):
            expert_idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_out = self.experts[e](expert_input)
                    output[mask] += w[mask] * expert_out
                    if self.training:
                        expert_outputs_for_tracking[e] = expert_out.detach()

        # Cross-expert knowledge transfer residual
        if self.enable_cross_transfer and self.training:
            transfer_signal = self.transfer_up(
                F.gelu(self.transfer_down(flat_x))
            )
            gate = torch.sigmoid(self.transfer_gate)
            output = output + gate * transfer_signal

        # Update compound tracker
        if self.training:
            self.compound_tracker.update(indices, weights, expert_outputs_for_tracking)

        # Compound regularization: penalize over-specialization
        compound_reg = self._compound_regularization()
        total_aux_loss = base_aux_loss + 0.001 * compound_reg

        return output.reshape(batch, seq_len, d), total_aux_loss

    def _compound_regularization(self) -> torch.Tensor:
        """Regularize to prevent expert collapse using compound statistics."""
        load = self.compound_tracker.expert_load_ema
        # Entropy-based: maximize entropy of expert load distribution
        load_prob = load / (load.sum() + 1e-8)
        entropy = -(load_prob * (load_prob + 1e-8).log()).sum()
        max_entropy = math.log(self.num_experts)
        # Loss is higher when entropy is low (unbalanced)
        return max_entropy - entropy

    def compound_learning_step(self, expert_rewards: Optional[torch.Tensor] = None):
        """
        External call to update compound learning state.
        Called after eval/validation to feed reward signals back.

        Args:
            expert_rewards: [num_experts] performance signal per expert
        """
        if expert_rewards is not None:
            self.compound_bias.update_compound_bias(expert_rewards)

    def get_expert_loads(self) -> torch.Tensor:
        """Return current expert load EMA for braid feedback."""
        return self.compound_tracker.expert_load_ema

    def get_compound_stats(self) -> Dict[str, Any]:
        """Return compound integration statistics."""
        stats = self.compound_tracker.get_stats()
        stats["transfer_gate"] = (
            torch.sigmoid(self.transfer_gate).item()
            if self.enable_cross_transfer
            else 0.0
        )
        stats["compound_bias_norm"] = (
            self.compound_bias.compound_bias.norm().item()
        )
        return stats

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_params(self) -> int:
        router_params = sum(p.numel() for p in self.router.parameters())
        expert_params = sum(p.numel() for p in self.experts[0].parameters())
        bias_params = sum(p.numel() for p in self.compound_bias.parameters())
        transfer_params = 0
        if self.enable_cross_transfer:
            transfer_params = sum(
                p.numel()
                for p in [self.transfer_down, self.transfer_up]
                for p in p.parameters()
            )
        return router_params + self.top_k * expert_params + bias_params + transfer_params


class CompoundLearningIntegration:
    """
    High-level integration that connects the NGVT CompoundLearningEngine
    with the OctoTetrahedral MoE model for end-to-end compound learning.

    Usage:
        model = OctoTetrahedralModel(config)
        integration = CompoundLearningIntegration(model)
        integration.record_forward_pass(inputs, outputs, metrics)
        integration.compound_update()
    """

    def __init__(self, model: nn.Module, learning_rate: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.step_count = 0
        self.history: List[Dict[str, Any]] = []
        self._compound_moe_layers = self._find_compound_layers()

    def _find_compound_layers(self) -> List[CompoundMoELayer]:
        """Find all CompoundMoELayer instances in the model."""
        layers = []
        for module in self.model.modules():
            if isinstance(module, CompoundMoELayer):
                layers.append(module)
        return layers

    def record_forward_pass(
        self,
        loss: float,
        accuracy: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record metrics from a forward pass for compound learning."""
        self.step_count += 1
        record = {
            "step": self.step_count,
            "loss": loss,
            "accuracy": accuracy,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.history.append(record)
        # Keep bounded history
        if len(self.history) > 10000:
            self.history = self.history[-5000:]

    def compound_update(self, eval_metrics: Optional[Dict[str, float]] = None):
        """
        Perform a compound learning update across all MoE layers.
        Call after evaluation to propagate reward signals.
        """
        if not self._compound_moe_layers:
            return

        # Compute expert rewards from recent history
        if eval_metrics and "per_expert_accuracy" in eval_metrics:
            rewards = torch.tensor(
                eval_metrics["per_expert_accuracy"], dtype=torch.float32
            )
        elif len(self.history) >= 10:
            # Derive reward from loss improvement trend
            recent = self.history[-10:]
            loss_trend = [r["loss"] for r in recent]
            improvement = loss_trend[0] - loss_trend[-1]
            # Uniform reward based on overall improvement
            num_experts = self._compound_moe_layers[0].num_experts
            rewards = torch.full((num_experts,), improvement)
        else:
            return

        for layer in self._compound_moe_layers:
            layer.compound_learning_step(rewards)

    def get_integration_report(self) -> Dict[str, Any]:
        """Generate a report of the compound integration state."""
        layer_stats = []
        for i, layer in enumerate(self._compound_moe_layers):
            stats = layer.get_compound_stats()
            stats["layer_index"] = i
            layer_stats.append(stats)

        return {
            "total_compound_layers": len(self._compound_moe_layers),
            "total_steps": self.step_count,
            "history_length": len(self.history),
            "layer_stats": layer_stats,
        }
