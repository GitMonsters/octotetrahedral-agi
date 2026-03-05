"""
Compound Braiding Module

Implements cross-limb information braiding where multiple processing limbs
exchange information mid-computation rather than running in isolation.

Standard approach: limbs run in parallel → outputs averaged
Braided approach:  limbs run → cross-attend to each other → gated combination

This lets spatial reasoning inform language, memory inform reasoning, etc.
The braid pattern is learned during training via cross-attention gates.

Compound MoE integration: braid gate patterns feed into MoE expert routing
via a braid_signal vector, and MoE expert specialization feeds back to
update braid combine weights.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class BraidCrossAttention(nn.Module):
    """Single cross-attention head for one limb attending to all others."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq, hidden] — the limb being updated
            context: [batch, seq, hidden] — concatenated other limbs' outputs
            mask: optional attention mask
        Returns:
            [batch, seq, hidden] — braided output for this limb
        """
        B, S, H = query.shape

        q = self.q_proj(query).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        return self.out_proj(out)


class CompoundBraid(nn.Module):
    """
    Cross-limb braiding: each limb attends to all other limbs' outputs,
    then a learned gate controls how much braided info to mix in.

    Replaces naive averaging of parallel limb outputs with learned
    cross-pollination between complementary processing streams.
    """

    LIMB_NAMES = ['memory', 'spatial', 'language', 'metacognition',
                  'reasoning', 'perception']

    def __init__(
        self,
        hidden_dim: int,
        num_limbs: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        braid_strength: float = 0.3,
        moe_signal_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_limbs = num_limbs
        self.braid_strength = braid_strength
        self.moe_signal_dim = moe_signal_dim

        # Each limb gets its own cross-attention to attend to others
        self.cross_attns = nn.ModuleList([
            BraidCrossAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_limbs)
        ])

        # Per-limb gating: controls how much braided info mixes in
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_limbs)
        ])

        # Final combination: learned weights for combining braided limbs
        self.combine_weights = nn.Parameter(torch.ones(num_limbs) / num_limbs)

        # Phase angles per limb (QM-inspired: phase determines interference)
        # Initialized to 0 so braiding starts identical to pre-phase behavior
        self.phase_angles = nn.Parameter(torch.zeros(num_limbs))

        # Layer norm per limb after braiding
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_limbs)
        ])

        # Braid→MoE signal: project gate patterns into a routing hint vector
        if moe_signal_dim > 0:
            self.braid_to_moe = nn.Linear(num_limbs, moe_signal_dim, bias=False)
        else:
            self.braid_to_moe = None

    def _apply_phase_rotation(self, x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Apply phase rotation by treating pairs of hidden dims as complex numbers.
        
        QM insight: phase relationships between basis states determine
        constructive/destructive interference. This lets limbs cancel noise
        and reinforce signal through learned phase offsets.
        """
        B, S, H = x.shape
        x_pairs = x.view(B, S, H // 2, 2)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rotated = torch.stack([
            cos_a * x_pairs[..., 0] - sin_a * x_pairs[..., 1],
            sin_a * x_pairs[..., 0] + cos_a * x_pairs[..., 1],
        ], dim=-1)
        return rotated.view(B, S, H)

    def forward(
        self,
        limb_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        moe_expert_loads: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Braid limb outputs together via cross-attention.

        Args:
            limb_outputs: list of [batch, seq, hidden] tensors, one per limb
            attention_mask: optional [batch, seq] mask
            moe_expert_loads: optional [num_experts] expert load EMA from
                              CompoundMoELayer — feeds back to adjust braid weights

        Returns:
            combined: [batch, seq, hidden] — braided combination
            braid_info: dict with per-limb gate values, attention stats,
                        and braid_signal for MoE routing
        """
        assert len(limb_outputs) == self.num_limbs

        braided = []
        gate_values = []

        for i in range(self.num_limbs):
            query = limb_outputs[i]

            # Context = all OTHER limbs concatenated along seq dim
            others = [limb_outputs[j] for j in range(self.num_limbs) if j != i]
            context = torch.cat(others, dim=1)  # [batch, seq*(N-1), hidden]

            # Cross-attend: this limb reads from all others
            braided_out = self.cross_attns[i](query, context, mask=None)

            # Gate: how much braided info to mix in
            gate_input = torch.cat([query, braided_out], dim=-1)
            gate = self.gates[i](gate_input)  # [batch, seq, hidden]
            gate_values.append(gate.mean().detach())

            # Mix: original + gated braided info
            mixed = query + self.braid_strength * gate * braided_out
            mixed = self.layer_norms[i](mixed)
            braided.append(mixed)

        # MoE feedback: if expert loads are provided, modulate combine weights
        # This creates a closed loop: braid → MoE → braid
        if moe_expert_loads is not None and self.training:
            # Expert load entropy as a modulation signal
            load_prob = moe_expert_loads / (moe_expert_loads.sum() + 1e-8)
            load_entropy = -(load_prob * (load_prob + 1e-8).log()).sum()
            max_entropy = math.log(moe_expert_loads.shape[0])
            # Low entropy (expert collapse) → increase braid diversity
            # High entropy (balanced) → keep current braid weights
            diversity_boost = torch.clamp(1.0 - load_entropy / max(max_entropy, 1e-8), 0.0, 0.2)
            # Push combine_weights toward uniform when experts are collapsing
            uniform = torch.ones_like(self.combine_weights) / self.num_limbs
            effective_weights = F.softmax(
                self.combine_weights + diversity_boost * (uniform - self.combine_weights),
                dim=0,
            )
        else:
            effective_weights = F.softmax(self.combine_weights, dim=0)

        # Combine with learned weights and phase rotations
        combined = sum(
            w * self._apply_phase_rotation(out, self.phase_angles[i])
            for i, (w, out) in enumerate(zip(effective_weights, braided))
        )

        # Generate braid→MoE routing signal from gate pattern
        gate_vector = torch.stack(gate_values)  # [num_limbs]
        braid_signal = None
        if self.braid_to_moe is not None:
            braid_signal = self.braid_to_moe(gate_vector)  # [moe_signal_dim]

        braid_info = {
            'gate_values': {
                name: gv.item() for name, gv in
                zip(self.LIMB_NAMES[:self.num_limbs], gate_values)
            },
            'combine_weights': {
                name: w.item() for name, w in
                zip(self.LIMB_NAMES[:self.num_limbs], effective_weights)
            },
            'phase_angles': {
                name: self.phase_angles[i].item() for i, name in
                enumerate(self.LIMB_NAMES[:self.num_limbs])
            },
            'braid_signal': braid_signal,
        }

        return combined, braid_info
