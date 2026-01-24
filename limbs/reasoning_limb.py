"""
Reasoning Limb - Abstract reasoning and pattern processing
Inspired by octopus central brain processing

Biological insight:
- Octopus brain has multiple lobes specialized for different tasks
- Vertical lobe handles learning and memory
- Superior frontal lobe integrates sensory information
- Brachial lobe coordinates arm movements

Our implementation:
- Multi-head self-attention for pattern detection
- Reasoning state extraction
- Uncertainty quantification
- Intermediate representation refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .base_limb import BaseLimb


class ReasoningAttention(nn.Module):
    """
    Lightweight multi-head attention for reasoning.
    Used within the limb for pattern extraction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            attention_mask: Optional mask [batch, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        return out, attn_weights.mean(dim=1)  # Average over heads


class ReasoningLimb(BaseLimb):
    """
    Reasoning Limb for abstract pattern processing.
    
    Takes encoded features and performs:
    1. Pattern detection via attention
    2. Feature refinement via MLP
    3. Uncertainty estimation
    4. Reasoning state extraction
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        buffer_size: int = 100
    ):
        super().__init__(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            buffer_size=buffer_size,
            limb_name="reasoning"
        )
        
        # Reasoning attention
        self.reasoning_attention = ReasoningAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feature refinement MLP
        self.refine_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Normalization layers
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        
        # Confidence head (outputs scalar uncertainty estimate)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Reasoning state projector (for external queries)
        self.state_projector = nn.Linear(hidden_dim, hidden_dim)
        
        # Track attention patterns for interpretability
        self._last_attention_weights = None
    
    def process(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Process input through reasoning layers.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Refined features [batch, seq_len, hidden_dim]
        """
        # Self-attention with residual
        attn_out, attn_weights = self.reasoning_attention(x, attention_mask)
        self._last_attention_weights = attn_weights.detach()
        x = self.attn_norm(x + attn_out)
        
        # MLP refinement with residual
        mlp_out = self.refine_mlp(x)
        output = self.mlp_norm(x + mlp_out)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_confidence: bool = False,
        return_state: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor]]:
        """
        Forward pass through reasoning limb.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            return_confidence: Whether to return confidence score
            return_state: Whether to return reasoning state vector
            
        Returns:
            Tuple of (output, confidence, reasoning_state)
        """
        # Base transformation + LoRA
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        
        # Reasoning processing
        output = self.process(adapted, attention_mask=attention_mask, **kwargs)
        
        # Confidence estimation
        confidence = None
        if return_confidence:
            # Pool over sequence for confidence
            pooled = output.mean(dim=1)  # [batch, hidden_dim]
            conf_scores = self.confidence_head(pooled)  # [batch, 1]
            confidence = conf_scores.mean().item()
        
        # Reasoning state extraction
        reasoning_state = None
        if return_state:
            # Use [CLS]-like pooling (first token or mean)
            pooled = output.mean(dim=1)  # [batch, hidden_dim]
            reasoning_state = self.state_projector(pooled)
        
        return output, confidence, reasoning_state
    
    def estimate_confidence(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor
    ) -> float:
        """
        Estimate reasoning confidence using multiple signals.
        """
        # Base confidence from parent
        base_conf = super().estimate_confidence(input_tensor, output_tensor)
        
        # Attention-based confidence
        # High entropy attention = low confidence (not sure where to look)
        attn_conf = 0.5
        if self._last_attention_weights is not None:
            with torch.no_grad():
                attn = self._last_attention_weights
                # Compute attention entropy
                attn_entropy = -(attn * torch.log(attn + 1e-10)).sum(-1).mean()
                max_entropy = math.log(attn.size(-1))
                attn_conf = 1.0 - (attn_entropy / max_entropy).item()
        
        # Learned confidence head output
        learned_conf = 0.5
        if hasattr(self, '_last_learned_conf'):
            learned_conf = self._last_learned_conf
        
        # Combine signals
        return 0.4 * base_conf + 0.3 * attn_conf + 0.3 * learned_conf
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last attention weights for interpretability"""
        return self._last_attention_weights
    
    def extract_reasoning_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract compact reasoning state vector.
        
        Useful for:
        - Querying the model's "understanding"
        - Passing state to other modules
        - Checkpoint/resume reasoning
        """
        with torch.no_grad():
            _, _, state = self.forward(x, return_state=True)
            return state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get limb statistics"""
        stats = super().get_stats()
        
        # Add attention stats
        if self._last_attention_weights is not None:
            attn = self._last_attention_weights
            stats['attention_entropy'] = (
                -(attn * torch.log(attn + 1e-10)).sum(-1).mean().item()
            )
            stats['attention_max'] = attn.max().item()
        
        return stats


if __name__ == "__main__":
    print("Testing ReasoningLimb...")
    
    # Create limb
    limb = ReasoningLimb(
        hidden_dim=256,
        num_heads=4,
        dropout=0.1
    )
    
    # Test input
    batch_size = 2
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 256)
    
    # Forward pass
    output, confidence, state = limb(
        x,
        return_confidence=True,
        return_state=True
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Reasoning state shape: {state.shape}")
    
    # Test attention weights
    attn = limb.get_attention_weights()
    print(f"Attention weights shape: {attn.shape}")
    
    # Test with mask
    mask = torch.ones(batch_size, seq_len)
    mask[:, 15:] = 0  # Mask last 5 tokens
    output_masked, _, _ = limb(x, attention_mask=mask)
    print(f"Masked output shape: {output_masked.shape}")
    
    # Stats
    stats = limb.get_stats()
    print(f"\nLimb stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Parameter count
    total_params = sum(p.numel() for p in limb.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nReasoningLimb tests passed!")
