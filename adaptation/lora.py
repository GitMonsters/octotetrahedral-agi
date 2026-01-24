"""
LoRA (Low-Rank Adaptation) Module
Parameter-efficient fine-tuning inspired by octopus RNA editing

Key insight from octopus biology:
- RNA editing changes specific sites without altering DNA
- LoRA changes specific weight matrices without full retraining
- Both enable rapid, reversible adaptation

LoRA decomposes weight updates as: W' = W + BA
where B ∈ R^{d×r} and A ∈ R^{r×k}, with r << min(d, k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for parameter-efficient fine-tuning.
    
    Instead of updating all parameters of a weight matrix W ∈ R^{d×k},
    we learn low-rank matrices A ∈ R^{r×k} and B ∈ R^{d×r} such that:
    
    h = Wx + BAx
    
    where r << min(d, k) is the rank (typically 4-16).
    
    This reduces trainable parameters from d*k to r*(d+k).
    For d=k=256 and r=4: 65,536 → 2,048 (32x reduction)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        merge_weights: bool = False
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False
        
        # Low-rank matrices
        # A is initialized with Kaiming uniform, B with zeros
        # This ensures ΔW = BA = 0 at initialization
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Optional dropout on LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize A with Kaiming uniform, B with zeros"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the low-rank weight update ΔW = BA * scaling
        
        Returns:
            Delta weight matrix [out_features, in_features]
        """
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def forward(
        self, 
        x: torch.Tensor, 
        base_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional base weight.
        
        If base_weight is provided: output = x @ (W + ΔW)^T
        If base_weight is None: output = x @ ΔW^T (just the adaptation)
        
        Args:
            x: Input tensor [..., in_features]
            base_weight: Optional base weight matrix [out_features, in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # Compute LoRA contribution: x @ A^T @ B^T * scaling
        lora_output = self.dropout(x)
        lora_output = F.linear(lora_output, self.lora_A)  # [..., rank]
        lora_output = F.linear(lora_output, self.lora_B)  # [..., out_features]
        lora_output = lora_output * self.scaling
        
        if base_weight is not None:
            # Add to base transformation
            base_output = F.linear(x, base_weight)
            return base_output + lora_output
        else:
            return lora_output
    
    def merge(self, base_weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA weights into base weight for efficient inference.
        
        Args:
            base_weight: Base weight matrix [out_features, in_features]
            
        Returns:
            Merged weight matrix [out_features, in_features]
        """
        return base_weight + self.get_delta_weight()
    
    def get_num_params(self) -> int:
        """Return number of trainable LoRA parameters"""
        return self.lora_A.numel() + self.lora_B.numel()


class LoRALinear(nn.Module):
    """
    Linear layer with integrated LoRA adaptation.
    
    Combines a frozen base linear layer with a trainable LoRA adapter.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True
    ):
        super().__init__()
        
        # Base linear layer
        self.base = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA adapter
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Optionally freeze base weights
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base + LoRA"""
        base_out = self.base(x)
        lora_out = self.lora(x)
        return base_out + lora_out
    
    def get_delta_weight(self) -> torch.Tensor:
        """Get the LoRA weight delta"""
        return self.lora.get_delta_weight()
    
    def merge_and_unload(self) -> nn.Linear:
        """Merge LoRA into base and return simple Linear"""
        merged_weight = self.lora.merge(self.base.weight.data)
        
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None
        )
        merged.weight.data = merged_weight
        if self.base.bias is not None:
            merged.bias.data = self.base.bias.data.clone()
        
        return merged


def apply_lora_to_linear(
    module: nn.Linear,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0
) -> LoRALinear:
    """
    Convert a standard Linear layer to LoRALinear.
    
    Args:
        module: Existing nn.Linear layer
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout rate
        
    Returns:
        LoRALinear with base weights copied from module
    """
    lora_linear = LoRALinear(
        in_features=module.in_features,
        out_features=module.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=module.bias is not None,
        freeze_base=True
    )
    
    # Copy base weights
    lora_linear.base.weight.data = module.weight.data.clone()
    if module.bias is not None:
        lora_linear.base.bias.data = module.bias.data.clone()
    
    return lora_linear


if __name__ == "__main__":
    # Test LoRA module
    print("Testing LoRA module...")
    
    in_features = 256
    out_features = 256
    rank = 4
    batch_size = 2
    seq_len = 10
    
    # Test LoRALayer
    lora = LoRALayer(in_features, out_features, rank=rank)
    x = torch.randn(batch_size, seq_len, in_features)
    base_weight = torch.randn(out_features, in_features)
    
    # Forward with base weight
    out = lora(x, base_weight=base_weight)
    print(f"LoRALayer output shape: {out.shape}")
    
    # Get delta weight
    delta = lora.get_delta_weight()
    print(f"Delta weight shape: {delta.shape}")
    print(f"Delta weight norm: {delta.norm().item():.6f}")
    
    # Test LoRALinear
    lora_linear = LoRALinear(in_features, out_features, rank=rank)
    out = lora_linear(x)
    print(f"LoRALinear output shape: {out.shape}")
    
    # Count parameters
    total_base = sum(p.numel() for p in lora_linear.base.parameters())
    total_lora = lora.get_num_params()
    print(f"\nBase parameters: {total_base:,}")
    print(f"LoRA parameters: {total_lora:,}")
    print(f"Reduction: {total_base / total_lora:.1f}x")
    
    # Test merge
    merged = lora_linear.merge_and_unload()
    out_merged = merged(x)
    diff = (out - out_merged).abs().max()
    print(f"\nMerge difference: {diff.item():.6f}")
    
    print("\nAll LoRA tests passed!")
