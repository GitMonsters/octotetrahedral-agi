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
from typing import Dict, List, Tuple, Optional, Union


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
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query: [batch, seq, hidden] — the limb being updated
            context: [batch, seq, hidden] — concatenated other limbs' outputs
            mask: optional attention mask over context tokens
            return_attention: whether to also return normalized attention weights
        Returns:
            [batch, seq, hidden] braided output for this limb, and optionally
            [batch, heads, query_seq, context_seq] attention weights
        """
        B, S, H = query.shape

        q = self.q_proj(query).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn_weights)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        out = self.out_proj(out)
        if return_attention:
            return out, attn_weights
        return out


class CompoundBraid(nn.Module):
    """
    Cross-limb braiding: each limb attends to all other limbs' outputs,
    then a learned gate controls how much braided info to mix in.

    Replaces naive averaging of parallel limb outputs with learned
    cross-pollination between complementary processing streams.

    CodeGen-specific braid patterns:
    - CodeGen → Spatial: torus position shapes geometric code structure.
    - Memory → CodeGen: past successful code patterns steer generation.
    - Reasoning → CodeGen: logical constraints validate generated code.
    - MetaCognition → CodeGen: self-critique refactors and tightens output.
    """

    LIMB_NAMES = [
        'memory', 'spatial', 'language', 'metacognition',
        'reasoning', 'perception', 'visualization', 'imagination',
        'codegen', 'empathy', 'emotion', 'ethics',
    ]

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
        if num_limbs < 2:
            raise ValueError("CompoundBraid requires at least two limbs to braid")
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for phase rotation")

        self.hidden_dim = hidden_dim
        self.num_limbs = num_limbs
        self.braid_strength = braid_strength
        self.moe_signal_dim = moe_signal_dim
        self.limb_names = self._resolve_limb_names(num_limbs)
        self.codegen_index = self.limb_names.index('codegen') if 'codegen' in self.limb_names else None

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

    @classmethod
    def _resolve_limb_names(cls, num_limbs: int) -> List[str]:
        if num_limbs <= len(cls.LIMB_NAMES):
            return cls.LIMB_NAMES[:num_limbs]
        extra_names = [f'stream_{i}' for i in range(num_limbs - len(cls.LIMB_NAMES))]
        return cls.LIMB_NAMES + extra_names

    @staticmethod
    def _summarize_attention(
        attention_weights: torch.Tensor,
        source_names: List[str],
        source_lengths: List[int],
    ) -> Dict[str, float]:
        token_weights = attention_weights.mean(dim=(0, 1, 2))
        summary: Dict[str, float] = {}
        start = 0
        for name, length in zip(source_names, source_lengths):
            end = start + length
            summary[name] = token_weights[start:end].sum().item()
            start = end
        total = sum(summary.values())
        if total > 0:
            summary = {name: value / total for name, value in summary.items()}
        return summary

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
        attention_stats = {}

        for i in range(self.num_limbs):
            query = limb_outputs[i]
            other_indices = [j for j in range(self.num_limbs) if j != i]

            # Context = all OTHER limbs concatenated along seq dim
            others = [limb_outputs[j] for j in other_indices]
            context = torch.cat(others, dim=1)  # [batch, seq*(N-1), hidden]
            context_mask = None
            if attention_mask is not None:
                context_mask = torch.cat([attention_mask for _ in other_indices], dim=1)

            # Cross-attend: this limb reads from all others
            braided_out, attn_weights = self.cross_attns[i](
                query,
                context,
                mask=context_mask,
                return_attention=True,
            )
            attention_stats[self.limb_names[i]] = self._summarize_attention(
                attn_weights.detach(),
                [self.limb_names[j] for j in other_indices],
                [limb_outputs[j].shape[1] for j in other_indices],
            )

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
                name: gv.item() for name, gv in zip(self.limb_names, gate_values)
            },
            'combine_weights': {
                name: w.item() for name, w in zip(self.limb_names, effective_weights)
            },
            'phase_angles': {
                name: self.phase_angles[i].item() for i, name in enumerate(self.limb_names)
            },
            'attention_weights': attention_stats,
            'braid_signal': braid_signal,
        }

        return combined, braid_info
