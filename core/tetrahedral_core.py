"""
Tetrahedral Core Module
Main transformer-based reasoning hub with tetrahedral geometry

This is the central processing unit that:
1. Receives embedded input from perception limb
2. Processes through geometry-aware transformer layers
3. Integrates with working memory
4. Produces reasoning state for downstream limbs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, Tuple, Dict, Any

from .tetrahedral_geometry import TetrahedralGeometry
from .tetrahedral_attention import TetrahedralAttention
from .moe import MoELayer
from .compound_moe import CompoundMoELayer


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.
    Standard transformer FFN: Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        ffn_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TetrahedralTransformerLayer(nn.Module):
    """
    Single transformer layer with tetrahedral attention.
    
    Structure:
    1. LayerNorm -> Tetrahedral Attention -> Residual
    2. LayerNorm -> FFN (dense or MoE) -> Residual
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_geometric_bias: bool = True,
        moe_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.use_moe = moe_config is not None and moe_config.get("enabled", False)
        self.use_compound_moe = self.use_moe and moe_config.get("compound_enabled", False)
        
        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = TetrahedralAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_geometric_bias=use_geometric_bias
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        if self.use_compound_moe:
            self.ffn = CompoundMoELayer(
                hidden_dim=hidden_dim,
                ffn_dim=moe_config.get("expert_ffn_dim", ffn_dim),
                num_experts=moe_config.get("num_experts", 64),
                top_k=moe_config.get("top_k", 8),
                dropout=dropout,
                jitter_noise=moe_config.get("jitter_noise", 0.01),
                compound_bias_scale=moe_config.get("compound_bias_scale", 0.1),
                enable_cross_transfer=moe_config.get("enable_cross_transfer", True),
            )
        elif self.use_moe:
            self.ffn = MoELayer(
                hidden_dim=hidden_dim,
                ffn_dim=moe_config.get("expert_ffn_dim", ffn_dim),
                num_experts=moe_config.get("num_experts", 64),
                top_k=moe_config.get("top_k", 8),
                dropout=dropout,
                jitter_noise=moe_config.get("jitter_noise", 0.01),
            )
        else:
            self.ffn = FeedForward(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        geometric_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_gates: Optional[torch.Tensor] = None,
        braid_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer layer.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            geometric_bias: Geometric attention bias [seq_len, seq_len]
            attention_mask: Optional attention mask
            head_gates: Optional head gating from RNA editing
            braid_signal: Optional [num_experts] routing hint from CompoundBraid
            
        Returns:
            Tuple of (output tensor, auxiliary MoE loss)
        """
        # Attention block with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(
            x, 
            geometric_bias=geometric_bias,
            attention_mask=attention_mask,
            head_gates=head_gates
        )
        x = residual + self.dropout(x)
        
        # FFN block with residual
        residual = x
        x = self.norm2(x)
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.use_compound_moe:
            x, aux_loss = self.ffn(x, braid_signal=braid_signal)
        elif self.use_moe:
            x, aux_loss = self.ffn(x)
        else:
            x = self.ffn(x)
        x = residual + x
        
        return x, aux_loss


class TetrahedralCore(nn.Module):
    """
    Central reasoning hub based on tetrahedral geometry.
    
    This module:
    1. Maintains the 64-point tetrahedral structure
    2. Processes input through multiple transformer layers
    3. Uses geometric bias in attention
    4. Outputs reasoning state for downstream processing
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        device: str = 'cpu',
        moe_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device
        
        # Initialize tetrahedral geometry
        self.geometry = TetrahedralGeometry(device=device)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TetrahedralTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_geometric_bias=True,
                moe_config=moe_config,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Projection to map sequence to fixed-size reasoning state
        # This allows variable-length input to produce consistent output
        self.reasoning_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable query for extracting reasoning state
        self.reasoning_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Gradient checkpointing flag (set externally to save memory)
        self.gradient_checkpointing = False
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_gates: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
        braid_signal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process input through tetrahedral reasoning core.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch, seq_len]
            head_gates: Optional per-head gating from RNA editing
            return_all_layers: Whether to return intermediate layer outputs
            braid_signal: Optional [num_experts] routing hint from CompoundBraid
            
        Returns:
            Dictionary containing:
                - 'hidden_states': Final hidden states [batch, seq_len, hidden_dim]
                - 'reasoning_state': Aggregated reasoning state [batch, hidden_dim]
                - 'layer_outputs': (optional) List of intermediate outputs
        """
        batch_size, seq_len, _ = x.shape
        
        # Get geometric bias from tetrahedral structure
        _, _, geometric_bias = self.geometry()
        
        layer_outputs = []
        total_aux_loss = torch.tensor(0.0, device=x.device)
        
        # Process through transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, aux_loss = grad_checkpoint(
                    layer, x, geometric_bias, attention_mask, head_gates,
                    use_reentrant=False
                )
            else:
                x, aux_loss = layer(
                    x,
                    geometric_bias=geometric_bias,
                    attention_mask=attention_mask,
                    head_gates=head_gates,
                    braid_signal=braid_signal,
                )
            total_aux_loss = total_aux_loss + aux_loss
            if return_all_layers:
                layer_outputs.append(x)
        
        # Final normalization
        hidden_states = self.final_norm(x)
        
        # Extract reasoning state via attention pooling
        # Use learnable query to attend over sequence
        query = self.reasoning_query.expand(batch_size, -1, -1)  # [batch, 1, hidden]
        
        # Simple attention: query attends to all hidden states
        attn_scores = torch.matmul(query, hidden_states.transpose(-2, -1))  # [batch, 1, seq_len]
        attn_scores = attn_scores / (self.hidden_dim ** 0.5)
        
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask.unsqueeze(1) == 0, 
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, 1, seq_len]
        reasoning_state = torch.matmul(attn_weights, hidden_states)  # [batch, 1, hidden]
        reasoning_state = reasoning_state.squeeze(1)  # [batch, hidden]
        
        # Project reasoning state
        reasoning_state = self.reasoning_projection(reasoning_state)
        
        output = {
            'hidden_states': hidden_states,
            'reasoning_state': reasoning_state,
            'aux_loss': total_aux_loss,
        }
        
        if return_all_layers:
            output['layer_outputs'] = layer_outputs
        
        return output
    
    def get_geometric_info(self) -> Dict[str, Any]:
        """Return information about the tetrahedral geometry"""
        return self.geometry.visualize_info()


if __name__ == "__main__":
    # Test the core module
    print("Testing TetrahedralCore...")
    
    batch_size = 2
    seq_len = 64
    hidden_dim = 256
    num_layers = 3
    num_heads = 8
    
    # Create core
    core = TetrahedralCore(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Forward pass
    output = core(x, return_all_layers=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Hidden states shape: {output['hidden_states'].shape}")
    print(f"Reasoning state shape: {output['reasoning_state'].shape}")
    print(f"Number of layer outputs: {len(output['layer_outputs'])}")
    
    # Test with head gating
    head_gates = torch.sigmoid(torch.randn(num_heads))
    output_gated = core(x, head_gates=head_gates)
    print(f"Gated reasoning state shape: {output_gated['reasoning_state'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in core.parameters())
    trainable_params = sum(p.numel() for p in core.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nAll core tests passed!")
