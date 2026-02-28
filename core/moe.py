"""
Mixture-of-Experts (MoE) module for OctoTetrahedral AGI.

Implements top-k expert routing with load-balancing loss,
enabling scaling to 1.72T total parameters with ~226B active per token.

Architecture:
    - Router: learned gating network selecting top-k experts per token
    - Expert FFN: independent feed-forward networks (SwiGLU activation)
    - Load balancing: auxiliary loss to prevent expert collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ExpertFFN(nn.Module):
    """Single expert feed-forward network with SwiGLU activation."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(dropout(silu(gate(x)) * up(x)))
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


class MoERouter(nn.Module):
    """Top-k expert routing with auxiliary load-balancing loss."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts.

        Args:
            x: [batch * seq_len, hidden_dim]

        Returns:
            expert_weights: [batch*seq, top_k] — normalized weights
            expert_indices: [batch*seq, top_k] — selected expert ids
            load_balance_loss: scalar auxiliary loss
        """
        # Optional jitter during training for exploration
        if self.training and self.jitter_noise > 0:
            x = x * (1.0 + torch.randn_like(x) * self.jitter_noise)

        logits = self.gate(x)  # [N, num_experts]
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Load-balancing loss (Switch Transformer style)
        # Encourage uniform expert utilization
        probs = F.softmax(logits, dim=-1)  # [N, E]
        # f_i = fraction of tokens routed to expert i
        tokens_per_expert = torch.zeros(
            self.num_experts, device=x.device, dtype=x.dtype
        )
        for k in range(self.top_k):
            tokens_per_expert.scatter_add_(
                0, top_k_indices[:, k], torch.ones_like(top_k_indices[:, k], dtype=x.dtype)
            )
        f = tokens_per_expert / max(x.shape[0], 1)
        # P_i = mean probability assigned to expert i
        P = probs.mean(dim=0)
        load_balance_loss = self.num_experts * (f * P).sum()

        return top_k_weights, top_k_indices, load_balance_loss


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer replacing standard FFN in transformer blocks.

    Each token is routed to top_k experts; outputs are weighted-summed.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int = 64,
        top_k: int = 8,
        dropout: float = 0.1,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoERouter(hidden_dim, num_experts, top_k, jitter_noise)
        self.experts = nn.ModuleList(
            [ExpertFFN(hidden_dim, ffn_dim, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, hidden_dim]

        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_loss: scalar load-balancing loss
        """
        batch, seq_len, d = x.shape
        flat_x = x.reshape(-1, d)  # [N, d]

        weights, indices, aux_loss = self.router(flat_x)  # [N, top_k]

        # Dispatch tokens to experts and combine
        output = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            expert_idx = indices[:, k]  # [N]
            w = weights[:, k].unsqueeze(-1)  # [N, 1]
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += w[mask] * expert_output

        return output.reshape(batch, seq_len, d), aux_loss

    def total_params(self) -> int:
        """Total parameters across all experts."""
        return sum(p.numel() for p in self.parameters())

    def active_params(self) -> int:
        """Parameters active per forward pass (top_k experts + router)."""
        router_params = sum(p.numel() for p in self.router.parameters())
        expert_params = sum(p.numel() for p in self.experts[0].parameters())
        return router_params + self.top_k * expert_params
