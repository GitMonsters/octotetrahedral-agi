"""Vision encoder for TranscendPlexity multi-modal AGI.

ViT-style patch encoder: image → patches → transformer → embeddings.
Supports arbitrary resolution via adaptive patching.
Output shape: [batch, num_patches, hidden_dim] for fusion with other modalities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, N, D]
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class VisionTransformerBlock(nn.Module):
    """Standard transformer block for vision."""

    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-style vision encoder for TranscendPlexity.

    Converts images of any size into a sequence of embeddings
    compatible with the compound braid fusion layer.

    Args:
        hidden_dim: Output embedding dimension (must match model hidden_dim)
        patch_size: Size of image patches (16 = 16×16 pixel patches)
        in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_patches: Maximum number of patches (for positional encoding)
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        num_layers: int = 4,
        num_heads: int = 8,
        max_patches: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, in_channels, hidden_dim)

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.modality_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Learnable 2D positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches + 1, hidden_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            VisionTransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Adaptive pooling to fixed sequence length for compound braid
        self.adaptive_proj = nn.Linear(hidden_dim, hidden_dim)

    def _interpolate_pos_embed(self, num_patches: int) -> torch.Tensor:
        """Interpolate positional embeddings for arbitrary patch counts."""
        pos = self.pos_embed[:, :num_patches + 1]
        if num_patches + 1 <= self.pos_embed.shape[1]:
            return pos
        # Interpolate for larger images
        pos_tokens = self.pos_embed[:, 1:]
        pos_tokens = pos_tokens.transpose(1, 2)  # [1, D, N]
        pos_tokens = F.interpolate(pos_tokens, size=num_patches, mode='linear', align_corners=False)
        pos_tokens = pos_tokens.transpose(1, 2)  # [1, N, D]
        return torch.cat([self.pos_embed[:, :1], pos_tokens], dim=1)

    def forward(
        self,
        images: torch.Tensor,
        target_seq_len: Optional[int] = None,
    ) -> dict:
        """Encode images into embeddings.

        Args:
            images: [B, C, H, W] tensor (any resolution, will be patched)
            target_seq_len: If set, pool/pad output to this sequence length
                           for alignment with text tokens

        Returns:
            dict with:
                'embeddings': [B, seq_len, hidden_dim] - patch embeddings
                'cls_token': [B, hidden_dim] - global image representation
                'num_patches': int - number of patches extracted
        """
        B = images.shape[0]

        # Extract patches
        x = self.patch_embed(images)  # [B, N, D]
        num_patches = x.shape[1]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, N+1, D]

        # Add positional encoding
        pos = self._interpolate_pos_embed(num_patches)
        x = x + pos[:, :x.shape[1]]

        # Add modality token (signals "this is vision" to the compound braid)
        x = x + self.modality_token

        # Transformer blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Split CLS and patch tokens
        cls_out = x[:, 0]  # [B, D]
        patch_out = x[:, 1:]  # [B, N, D]

        # Adaptive pooling to target sequence length
        if target_seq_len is not None and patch_out.shape[1] != target_seq_len:
            patch_out = patch_out.transpose(1, 2)  # [B, D, N]
            patch_out = F.adaptive_avg_pool1d(patch_out, target_seq_len)
            patch_out = patch_out.transpose(1, 2)  # [B, target_seq_len, D]

        patch_out = self.adaptive_proj(patch_out)

        return {
            'embeddings': patch_out,
            'cls_token': cls_out,
            'num_patches': num_patches,
        }

    def encode_grid(self, grid) -> dict:
        """Encode an ARC-style grid as a 'visual' input.

        Converts a grid (values 0-9) into a pseudo-image by mapping
        colors to RGB, then encoding.

        Args:
            grid: [B, H, W] tensor, or a 2D Python list [[int, ...], ...]

        Returns:
            Same as forward()
        """
        # Convert list to tensor if needed
        if isinstance(grid, list):
            grid = torch.tensor(grid, dtype=torch.long)
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)  # Add batch dim

        device = next(self.parameters()).device
        grid = grid.to(device)

        # ARC color palette (matching the standard visualization)
        palette = torch.tensor([
            [0, 0, 0],        # 0: black
            [0, 116, 217],    # 1: blue
            [255, 65, 54],    # 2: red
            [46, 204, 64],    # 3: green
            [255, 220, 0],    # 4: yellow
            [170, 170, 170],  # 5: gray
            [240, 18, 190],   # 6: magenta
            [255, 133, 27],   # 7: orange
            [127, 219, 255],  # 8: azure
            [135, 12, 37],    # 9: maroon
        ], dtype=torch.float32, device=device) / 255.0

        # Map grid values to RGB
        grid_clamped = grid.clamp(0, 9).long()
        rgb = palette[grid_clamped]  # [B, H, W, 3]
        rgb = rgb.permute(0, 3, 1, 2)  # [B, 3, H, W]

        # Upscale small grids to minimum patch size
        if rgb.shape[2] < self.patch_size or rgb.shape[3] < self.patch_size:
            scale = max(self.patch_size // min(rgb.shape[2], rgb.shape[3]) + 1, 2)
            rgb = F.interpolate(rgb, scale_factor=scale, mode='nearest')

        return self.forward(rgb)
