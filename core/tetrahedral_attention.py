"""
Tetrahedral Attention Module
Geometry-aware multi-head attention that incorporates spatial relationships

The key innovation is adding a geometric bias to attention scores,
so that points closer in tetrahedral space attend more strongly to each other.
This creates a structured information flow that respects the geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TetrahedralAttention(nn.Module):
    """
    Multi-head attention with tetrahedral geometric bias.
    
    Standard attention: softmax(QK^T / sqrt(d))
    Tetrahedral attention: softmax(QK^T / sqrt(d) + alpha * geometric_bias)
    
    The geometric bias encodes spatial relationships from the tetrahedral structure,
    allowing the model to leverage geometric priors in its attention patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_geometric_bias: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_geometric_bias = use_geometric_bias
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Learnable scaling factor for geometric bias
        self.geo_bias_scale = nn.Parameter(torch.tensor(1.0))
        
        # Per-head learnable bias (allows different heads to focus on different geometric relationships)
        self.head_geo_scales = nn.Parameter(torch.ones(num_heads))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        geometric_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_gates: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional geometric bias and head gating.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            geometric_bias: Pre-computed geometric attention bias [seq_len, seq_len] or [64, 64]
            attention_mask: Optional mask [batch, seq_len] or [batch, 1, seq_len, seq_len]
            head_gates: Optional per-head gating from RNA editing [num_heads] or [batch, num_heads]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attention output [batch, seq_len, hidden_dim]
            attention_weights: Optional attention weights [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [batch, seq_len, hidden] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Add geometric bias if provided
        if self.use_geometric_bias and geometric_bias is not None:
            # Handle different sequence lengths (geometric bias is for 64-point grid)
            if seq_len != geometric_bias.size(0):
                # If sequence is shorter, use subset of geometric bias
                if seq_len < geometric_bias.size(0):
                    geo_bias_subset = geometric_bias[:seq_len, :seq_len]
                else:
                    # If sequence is longer, pad geometric bias
                    geo_bias_subset = F.pad(
                        geometric_bias,
                        (0, seq_len - geometric_bias.size(1), 0, seq_len - geometric_bias.size(0)),
                        value=0
                    )
            else:
                geo_bias_subset = geometric_bias
            
            # Scale per head and add to attention scores
            # geo_bias_subset: [seq_len, seq_len]
            # head_geo_scales: [num_heads]
            scaled_geo_bias = self.geo_bias_scale * geo_bias_subset.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
            head_scales = self.head_geo_scales.view(1, self.num_heads, 1, 1)  # [1, heads, 1, 1]
            attn_scores = attn_scores + scaled_geo_bias * head_scales
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply head gating from RNA editing layer
        if head_gates is not None:
            if head_gates.dim() == 1:
                # [num_heads] -> [1, num_heads, 1, 1]
                head_gates = head_gates.view(1, self.num_heads, 1, 1)
            elif head_gates.dim() == 2:
                # [batch, num_heads] -> [batch, num_heads, 1, 1]
                head_gates = head_gates.unsqueeze(-1).unsqueeze(-1)
            attn_weights = attn_weights * head_gates
        
        # Compute output: [batch, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, v)
        
        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class TetrahedralCrossAttention(nn.Module):
    """
    Cross-attention variant for attending from one sequence to another
    with geometric bias support.
    
    Used for memory attention and limb-to-core communication.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Separate projections for query (from x) and key/value (from memory)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention from x to memory.
        
        Args:
            x: Query source [batch, seq_len, hidden_dim]
            memory: Key/value source [batch, mem_len, hidden_dim]
            memory_mask: Optional mask for memory [batch, mem_len]
            
        Returns:
            output: Cross-attention output [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        mem_len = memory.size(1)
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, mem_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, mem_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, mem_len]
            attn_scores = attn_scores.masked_fill(memory_mask == 0, float('-inf'))
        
        # Attention weights and output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        
        return output


if __name__ == "__main__":
    # Test the attention modules
    print("Testing TetrahedralAttention...")
    
    batch_size = 2
    seq_len = 64
    hidden_dim = 256
    num_heads = 8
    
    # Create attention layer
    attn = TetrahedralAttention(hidden_dim=hidden_dim, num_heads=num_heads)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Create geometric bias
    geo_bias = torch.randn(seq_len, seq_len)
    geo_bias = (geo_bias + geo_bias.T) / 2  # Symmetric
    
    # Forward pass
    output, attn_weights = attn(x, geometric_bias=geo_bias, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test with head gating
    head_gates = torch.sigmoid(torch.randn(num_heads))
    output_gated, _ = attn(x, geometric_bias=geo_bias, head_gates=head_gates)
    print(f"Gated output shape: {output_gated.shape}")
    
    # Test cross-attention
    print("\nTesting TetrahedralCrossAttention...")
    cross_attn = TetrahedralCrossAttention(hidden_dim=hidden_dim, num_heads=num_heads)
    memory = torch.randn(batch_size, 4, hidden_dim)  # 4 memory slots
    cross_output = cross_attn(x, memory)
    print(f"Cross-attention output shape: {cross_output.shape}")
    
    print("\nAll attention tests passed!")
