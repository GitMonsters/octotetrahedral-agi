"""
Spatial Limb - Spatial reasoning and geometric understanding
Inspired by octopus spatial navigation and arm coordination

Biological insight:
- Octopuses navigate complex 3D environments
- Coordinate 8 arms in 3D space simultaneously
- Show object manipulation skills
- Demonstrate spatial memory for dens/hunting grounds

Our implementation:
- 3D coordinate encoding
- Spatial attention mechanisms
- Geometric transformation understanding
- Grid/lattice reasoning (for ARC)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

from .base_limb import BaseLimb


class SpatialPositionEncoding(nn.Module):
    """
    Encodes spatial positions in 2D/3D space.
    Supports both continuous coordinates and discrete grids.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_grid_size: int = 30,
        num_dims: int = 2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        self.num_dims = num_dims
        
        # Learnable position embeddings for discrete grids
        self.x_embed = nn.Embedding(max_grid_size, hidden_dim // num_dims)
        self.y_embed = nn.Embedding(max_grid_size, hidden_dim // num_dims)
        
        if num_dims == 3:
            self.z_embed = nn.Embedding(max_grid_size, hidden_dim // num_dims)
        
        # Continuous position encoding (sinusoidal)
        self.freq_bands = nn.Parameter(
            torch.linspace(1, max_grid_size // 2, hidden_dim // (2 * num_dims)),
            requires_grad=False
        )
    
    def encode_discrete(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode discrete grid positions."""
        x_enc = self.x_embed(x.clamp(0, self.max_grid_size - 1))
        y_enc = self.y_embed(y.clamp(0, self.max_grid_size - 1))
        
        if self.num_dims == 3 and z is not None:
            z_enc = self.z_embed(z.clamp(0, self.max_grid_size - 1))
            return torch.cat([x_enc, y_enc, z_enc], dim=-1)
        
        # Pad to full hidden dim
        padding = torch.zeros(*x_enc.shape[:-1], self.hidden_dim - x_enc.size(-1) - y_enc.size(-1), device=x_enc.device)
        return torch.cat([x_enc, y_enc, padding], dim=-1)
    
    def encode_continuous(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous coordinates using sinusoidal encoding.
        
        Args:
            coords: [batch, num_points, num_dims] in range [0, 1]
        """
        batch_size, num_points, dims = coords.shape
        
        # Expand for frequency bands
        coords_exp = coords.unsqueeze(-1)  # [batch, points, dims, 1]
        freqs = self.freq_bands.view(1, 1, 1, -1)  # [1, 1, 1, bands]
        
        # Sinusoidal encoding
        sin_enc = torch.sin(2 * math.pi * coords_exp * freqs)
        cos_enc = torch.cos(2 * math.pi * coords_exp * freqs)
        
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        encoding = encoding.reshape(batch_size, num_points, -1)
        
        # Project to hidden dim
        if encoding.size(-1) != self.hidden_dim:
            padding = torch.zeros(
                batch_size, num_points,
                self.hidden_dim - encoding.size(-1),
                device=coords.device
            )
            encoding = torch.cat([encoding, padding], dim=-1)
        
        return encoding[:, :, :self.hidden_dim]


class SpatialAttention(nn.Module):
    """
    Attention mechanism that respects spatial locality.
    Nearby positions attend more strongly to each other.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        locality_sigma: float = 2.0
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.locality_sigma = locality_sigma
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Features [batch, seq_len, hidden_dim]
            positions: Spatial positions [batch, seq_len, 2 or 3]
            attention_mask: Optional mask
        """
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Content-based attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add spatial locality bias if positions provided
        if positions is not None:
            # Compute pairwise distances
            pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch, seq, seq, dims]
            distances = (pos_diff ** 2).sum(-1).sqrt()  # [batch, seq, seq]
            
            # Gaussian locality bias
            locality_bias = torch.exp(-distances ** 2 / (2 * self.locality_sigma ** 2))
            locality_bias = locality_bias.unsqueeze(1)  # [batch, 1, seq, seq]
            
            attn = attn + locality_bias.log()
        
        if attention_mask is not None:
            attn = attn.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        return out, attn_weights.mean(dim=1)


class GeometricTransformPredictor(nn.Module):
    """
    Predicts geometric transformations between input/output grids.
    Useful for ARC-style reasoning.
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Transform type classifier
        self.transform_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8)  # rotation, flip, scale, translate, etc.
        )
        
        # Transform parameter predictor
        self.param_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6)  # 6 DOF for 2D affine
        )
        
        self.transform_names = [
            'identity', 'rotate_90', 'rotate_180', 'rotate_270',
            'flip_h', 'flip_v', 'scale', 'translate'
        ]
    
    def forward(
        self,
        input_repr: torch.Tensor,
        output_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Predict transformation from input to output.
        
        Args:
            input_repr: [batch, hidden_dim]
            output_repr: [batch, hidden_dim]
            
        Returns:
            (transform_logits, transform_params, predicted_transforms)
        """
        combined = torch.cat([input_repr, output_repr], dim=-1)
        
        transform_logits = self.transform_classifier(combined)
        transform_params = self.param_predictor(combined)
        
        # Get predicted transforms
        predicted_indices = transform_logits.argmax(dim=-1)
        predicted_transforms = [self.transform_names[i] for i in predicted_indices.tolist()]
        
        return transform_logits, transform_params, predicted_transforms


class SpatialLimb(BaseLimb):
    """
    Spatial Limb for spatial reasoning and geometric understanding.
    
    Capabilities:
    1. Grid/lattice representation
    2. Spatial position encoding
    3. Geometric transformation reasoning
    4. Locality-aware attention
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        max_grid_size: int = 30,
        num_heads: int = 4,
        num_layers: int = 2,
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
            limb_name="spatial"
        )
        
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        
        # Spatial position encoding
        self.pos_encoding = SpatialPositionEncoding(
            hidden_dim=hidden_dim,
            max_grid_size=max_grid_size
        )
        
        # Spatial attention layers
        self.spatial_attns = nn.ModuleList([
            SpatialAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # FFN layers
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Norms
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Geometric transform predictor
        self.transform_predictor = GeometricTransformPredictor(hidden_dim)
        
        # Grid encoder (for ARC-style grids)
        self.grid_encoder = nn.Sequential(
            nn.Linear(11, hidden_dim // 2),  # 11 colors in ARC (0-10)
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Stats
        self._grids_processed = 0
        self._transforms_predicted = 0
    
    def encode_grid(
        self,
        grid: torch.Tensor,
        add_positions: bool = True
    ) -> torch.Tensor:
        """
        Encode a 2D grid to sequence of features.
        
        Args:
            grid: [batch, height, width] with integer cell values
            add_positions: Whether to add position encodings
            
        Returns:
            [batch, height*width, hidden_dim]
        """
        batch_size, height, width = grid.shape
        
        # One-hot encode cell values
        grid_flat = grid.reshape(batch_size, -1)  # [batch, h*w]
        grid_onehot = F.one_hot(grid_flat.clamp(0, 10), num_classes=11).float()
        
        # Encode
        features = self.grid_encoder(grid_onehot)  # [batch, h*w, hidden]
        
        # Add position encodings
        if add_positions:
            y_coords = torch.arange(height, device=grid.device).repeat_interleave(width)
            x_coords = torch.arange(width, device=grid.device).repeat(height)
            
            pos_enc = self.pos_encoding.encode_discrete(
                x_coords.unsqueeze(0).expand(batch_size, -1),
                y_coords.unsqueeze(0).expand(batch_size, -1)
            )
            
            features = features + pos_enc
        
        self._grids_processed += batch_size
        return features
    
    def process(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Process spatial features.
        
        Args:
            x: Features [batch, seq_len, hidden_dim]
            positions: Optional positions [batch, seq_len, 2]
            attention_mask: Optional mask
            
        Returns:
            Processed features
        """
        # Spatial attention layers
        for attn, ffn, attn_norm, ffn_norm in zip(
            self.spatial_attns, self.ffns,
            self.attn_norms, self.ffn_norms
        ):
            # Attention with locality
            attn_out, _ = attn(x, positions=positions, attention_mask=attention_mask)
            x = attn_norm(x + attn_out)
            
            # FFN
            ffn_out = ffn(x)
            x = ffn_norm(x + ffn_out)
        
        # Output
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        return_confidence: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[Dict]]:
        """
        Forward pass through spatial limb.
        
        Args:
            x: Pre-computed features [batch, seq_len, hidden_dim]
               OR ignored if grid provided
            grid: Optional 2D grid [batch, height, width]
            positions: Optional position coordinates
            return_confidence: Whether to return confidence
        """
        # Encode grid if provided
        if grid is not None:
            x = self.encode_grid(grid)
        
        # Base transformation + LoRA
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        
        # Spatial processing
        output = self.process(adapted, positions=positions, **kwargs)
        
        # Confidence
        confidence = None
        if return_confidence:
            confidence = self.estimate_confidence(x, output)
        
        return output, confidence, None
    
    def predict_transform(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Predict transformation between input and output grids.
        
        Args:
            input_grid: [batch, h, w]
            output_grid: [batch, h, w]
            
        Returns:
            (transform_logits, transform_params, transform_names)
        """
        # Encode both grids
        input_features = self.encode_grid(input_grid)
        output_features = self.encode_grid(output_grid)
        
        # Pool to single vectors
        input_repr = input_features.mean(dim=1)
        output_repr = output_features.mean(dim=1)
        
        # Predict transform
        result = self.transform_predictor(input_repr, output_repr)
        self._transforms_predicted += input_grid.size(0)
        
        return result
    
    def reconstruct_grid(
        self,
        features: torch.Tensor,
        height: int,
        width: int,
        num_colors: int = 11
    ) -> torch.Tensor:
        """
        Reconstruct grid from features.
        
        Args:
            features: [batch, h*w, hidden_dim]
            height, width: Grid dimensions
            num_colors: Number of possible colors
            
        Returns:
            [batch, height, width] with predicted colors
        """
        batch_size = features.size(0)
        
        # Simple linear projection to color logits
        # In practice, would use a more sophisticated decoder
        color_proj = nn.Linear(self.hidden_dim, num_colors).to(features.device)
        color_logits = color_proj(features)  # [batch, h*w, colors]
        
        # Predict colors
        colors = color_logits.argmax(dim=-1)  # [batch, h*w]
        
        return colors.reshape(batch_size, height, width)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get spatial limb statistics."""
        stats = super().get_stats()
        stats.update({
            'max_grid_size': self.max_grid_size,
            'grids_processed': self._grids_processed,
            'transforms_predicted': self._transforms_predicted
        })
        return stats


if __name__ == "__main__":
    print("Testing SpatialLimb...")
    
    # Create limb
    limb = SpatialLimb(
        hidden_dim=256,
        max_grid_size=30,
        num_heads=4,
        num_layers=2
    )
    
    # Test with pre-computed features
    batch_size = 2
    seq_len = 25  # 5x5 grid
    x = torch.randn(batch_size, seq_len, 256)
    
    output, confidence, _ = limb(x, return_confidence=True)
    print(f"Feature input shape: {x.shape}")
    print(f"Feature output shape: {output.shape}")
    print(f"Confidence: {confidence:.4f}")
    
    # Test with grid input
    grid = torch.randint(0, 10, (batch_size, 5, 5))
    output_grid, _, _ = limb(x=None, grid=grid)
    print(f"\nGrid input shape: {grid.shape}")
    print(f"Grid output shape: {output_grid.shape}")
    
    # Test transform prediction
    input_grid = torch.randint(0, 10, (batch_size, 5, 5))
    output_grid = torch.randint(0, 10, (batch_size, 5, 5))
    
    logits, params, transforms = limb.predict_transform(input_grid, output_grid)
    print(f"\nTransform logits shape: {logits.shape}")
    print(f"Transform params shape: {params.shape}")
    print(f"Predicted transforms: {transforms}")
    
    # Stats
    stats = limb.get_stats()
    print(f"\nSpatial Limb stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Parameter count
    total_params = sum(p.numel() for p in limb.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nSpatialLimb tests passed!")
