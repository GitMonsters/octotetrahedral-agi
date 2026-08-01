"""
Recurrent-Depth Transformer (RDT)

Adaptive depth reasoning module that learns when to recurse deeper or exit.
Integrates with CompoundLoopController to provide intelligent early stopping
and dynamically modulates cross-limb communication intensity based on recursion depth.

Key outputs:
- depth_exit_logits: Probability of terminating recursion at this level
- uncertainty: Epistemic uncertainty (drives adaptive routing)
- routing_gates: Per-limb attention modulation for the CompoundBraid
- state_compression: Lossy summary for next recursion level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import math


class DepthAwareAttention(nn.Module):
    """
    Attention mechanism that's aware of recursion depth.
    Scales attention intensity based on depth level.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        depth: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            depth: Current recursion depth (0 = initial, higher = deeper)
            mask: Optional attention mask
        
        Returns:
            output: [batch, seq_len, hidden_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale factor depends on depth: deeper = more selective attention
        depth_scale = 1.0 / math.sqrt(1.0 + 0.1 * depth)
        base_scale = self.head_dim ** -0.5
        scale = base_scale * depth_scale
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        
        return out, attn_weights


class DepthTransformerLayer(nn.Module):
    """Single layer of depth-aware transformer."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = DepthAwareAttention(hidden_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, depth: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(self.norm1(x), depth=depth, mask=mask)
        x = x + attn_out
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class RecurrentDepthTransformer(nn.Module):
    """
    Recurrent transformer that learns adaptive depth for reasoning.
    
    Given a state representation, produces:
    1. exit_logits: Should we stop recursing?
    2. uncertainty: How uncertain are we about the exit decision?
    3. routing_gates: How intensely should limbs interact (braid gating)?
    4. state_compression: Compressed state for next recursion level
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        num_limbs: int = 14,  # 11 cognitive + vision + audio + embodiment
        max_depth: int = 8,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_limbs = num_limbs
        self.max_depth = max_depth
        
        # Compress input state to fixed size for depth decision
        self.state_compressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Transformer layers for depth reasoning
        self.transformer_layers = nn.ModuleList([
            DepthTransformerLayer(hidden_dim // 2, num_heads, ffn_dim // 2, dropout)
            for _ in range(num_layers)
        ])
        
        # Exit decision head: should we stop recursing?
        # Outputs logits for exit vs. continue
        self.exit_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2),  # [logits_exit, logits_continue]
        )
        
        # Uncertainty head: how confident are we?
        # Higher uncertainty → more exploration (deeper recursion)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        # Routing gates: per-limb attention modulation
        # Used by CompoundBraid to weight cross-limb communication
        self.routing_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_limbs),
        )
        
        # State compression for next level (reconstruction)
        self.state_reconstructor = nn.Linear(hidden_dim // 2, hidden_dim)
        
    def forward(
        self,
        state: torch.Tensor,
        depth: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: [batch, seq_len, hidden_dim] — current processing state
            depth: Current recursion depth (0 = top level)
            attention_mask: Optional mask for sequence positions
            return_components: Whether to return intermediate activations
        
        Returns:
            Dict with:
                - exit_logits: [batch, 2] — logits for (exit, continue)
                - exit_prob: [batch] — probability of exiting (0-1)
                - uncertainty: [batch] — epistemic uncertainty (0-1)
                - routing_gates: [batch, num_limbs] — per-limb gating (0-1)
                - state_reconstruction: [batch, seq_len, hidden_dim] — next level state
                - depth_reached: int — current depth
        """
        batch_size, seq_len, hidden_dim = state.shape
        assert hidden_dim == self.hidden_dim, f"Expected {self.hidden_dim}, got {hidden_dim}"
        
        # Compress state for depth reasoning
        compressed = self.state_compressor(state)  # [batch, seq_len, hidden_dim//2]
        
        # Pool to single vector for depth decision (mean over sequence)
        state_summary = compressed.mean(dim=1)  # [batch, hidden_dim//2]
        
        # Process through transformer layers with depth awareness
        x = compressed
        for layer in self.transformer_layers:
            x = layer(x, depth=depth, mask=attention_mask)
        
        # Re-pool after transformation
        processed_summary = x.mean(dim=1)  # [batch, hidden_dim//2]
        
        # Exit decision: should we recurse deeper or stop?
        exit_logits = self.exit_head(processed_summary)  # [batch, 2]
        exit_probs = F.softmax(exit_logits, dim=-1)
        exit_prob = exit_probs[:, 0]  # Probability of exiting
        
        # Uncertainty: how unsure are we about the exit decision?
        # Higher uncertainty (closer to 0.5) → more exploration
        uncertainty = self.uncertainty_head(processed_summary)  # [batch, 1]
        uncertainty = torch.sigmoid(uncertainty).squeeze(-1)  # [batch]
        
        # Routing gates: modulate braid intensity based on depth and state
        routing_logits = self.routing_head(processed_summary)  # [batch, num_limbs]
        routing_gates = torch.sigmoid(routing_logits)  # [batch, num_limbs]
        
        # Depth-aware routing: deeper levels get tighter coupling
        depth_factor = 1.0 + 0.2 * min(depth, self.max_depth) / self.max_depth
        routing_gates = routing_gates * depth_factor
        routing_gates = torch.clamp(routing_gates, 0.0, 1.0)
        
        # State reconstruction: prepare input for next recursion level
        state_reconstruction = self.state_reconstructor(processed_summary)  # [batch, hidden_dim]
        state_reconstruction = state_reconstruction.unsqueeze(1).expand(
            batch_size, seq_len, self.hidden_dim
        )  # [batch, seq_len, hidden_dim]
        
        # Halting probability (for ACT integration): similar to exit but for formal ACT
        # High exit_prob → high halting_prob (should stop processing)
        halting_prob = exit_prob  # Reuse exit probability for ACT halting
        
        result = {
            'exit_logits': exit_logits,  # [batch, 2]
            'exit_prob': exit_prob,  # [batch]
            'halting_prob': halting_prob,  # [batch] — for ACT integration
            'uncertainty': uncertainty,  # [batch]
            'routing_gates': routing_gates,  # [batch, num_limbs]
            'state_reconstruction': state_reconstruction,  # [batch, seq_len, hidden_dim]
            'depth_reached': depth,
        }
        
        if return_components:
            result.update({
                'compressed_state': compressed,
                'processed_summary': processed_summary,
                'exit_logits_detail': exit_logits,
            })
        
        return result
    
    def get_depth_loss(
        self,
        exit_probs: torch.Tensor,
        target_depth: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Regularize depth: encourage early exit when confident.
        
        Args:
            exit_probs: [batch] — probability of exiting at this level
            target_depth: Target average depth (None = no preference)
        
        Returns:
            Loss encouraging exit when appropriate
        """
        # Entropy of exit decision: lower entropy = more committed
        exit_entropy = -exit_probs * torch.log(exit_probs + 1e-10)
        exit_entropy -= (1 - exit_probs) * torch.log(1 - exit_probs + 1e-10)
        
        # Encourage decisive decisions (low entropy) when confident
        loss = exit_entropy.mean()
        
        return loss
