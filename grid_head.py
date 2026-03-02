"""
Grid Prediction Head for OctoTetrahedral AGI

Instead of autoregressive next-token prediction, this head directly predicts
entire output grids as (H, W) classifications over 10 ARC colors.

Architecture:
    Transformer hidden states [batch, seq_len, hidden_dim]
            ↓
    Pool to task representation [batch, hidden_dim]
            ↓
    Cross-attention: learned grid queries attend to context
            ↓
    Per-cell color classifier → [batch, max_H, max_W, 10]
    Dimension predictor → [batch, 2] (height, width)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


MAX_GRID_SIZE = 30  # ARC max grid dimension
NUM_COLORS = 10     # ARC colors 0-9


class GridPredictionHead(nn.Module):
    """Predicts ARC output grids from transformer hidden states."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 2,
                 num_heads: int = 4, max_grid: int = MAX_GRID_SIZE):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_grid = max_grid

        # Learned 2D position embeddings for each potential output cell
        self.row_embed = nn.Embedding(max_grid, hidden_dim // 2)
        self.col_embed = nn.Embedding(max_grid, hidden_dim // 2)

        # Cross-attention layers: grid queries attend to transformer context
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Per-cell color classifier
        self.color_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, NUM_COLORS)
        )

        # Dimension predictor: predicts output (height, width)
        self.dim_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_grid * 2)  # height logits + width logits
        )

    def _build_grid_queries(self, batch_size: int, device: torch.device,
                            dtype: torch.dtype) -> torch.Tensor:
        """Build 2D positional queries for all possible grid cells.

        Returns: [batch, max_grid * max_grid, hidden_dim]
        """
        rows = torch.arange(self.max_grid, device=device)
        cols = torch.arange(self.max_grid, device=device)
        row_emb = self.row_embed(rows)  # [max_grid, hidden_dim//2]
        col_emb = self.col_embed(cols)  # [max_grid, hidden_dim//2]

        # Create 2D grid of embeddings
        # row_emb: [H, 1, D//2] + col_emb: [1, W, D//2] → [H, W, D]
        grid_emb = torch.cat([
            row_emb.unsqueeze(1).expand(-1, self.max_grid, -1),
            col_emb.unsqueeze(0).expand(self.max_grid, -1, -1)
        ], dim=-1)  # [max_grid, max_grid, hidden_dim]

        grid_queries = grid_emb.reshape(1, self.max_grid * self.max_grid, self.hidden_dim)
        return grid_queries.expand(batch_size, -1, -1).to(dtype)

    def forward(self, hidden_states: torch.Tensor,
                target_h: Optional[int] = None,
                target_w: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim] from transformer
            target_h, target_w: if provided, only predict for this grid size

        Returns:
            grid_logits: [batch, max_grid, max_grid, 10] color logits
            dim_logits: [batch, 2, max_grid] height and width logits
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Build grid queries with 2D position embeddings
        grid_queries = self._build_grid_queries(batch_size, device, dtype)

        # Cross-attend: grid queries attend to transformer hidden states
        grid_features = self.decoder(
            grid_queries,  # [batch, H*W, hidden_dim]
            hidden_states  # [batch, seq_len, hidden_dim]
        )  # [batch, H*W, hidden_dim]

        # Predict colors per cell
        grid_logits = self.color_head(grid_features)  # [batch, H*W, 10]
        grid_logits = grid_logits.reshape(
            batch_size, self.max_grid, self.max_grid, NUM_COLORS
        )

        # Predict output dimensions from pooled context
        ctx_pooled = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        dim_raw = self.dim_head(ctx_pooled)  # [batch, max_grid * 2]
        dim_logits = dim_raw.reshape(batch_size, 2, self.max_grid)

        return {
            'grid_logits': grid_logits,  # [batch, H, W, 10]
            'dim_logits': dim_logits,    # [batch, 2, max_grid]
        }


def grid_loss(pred: Dict[str, torch.Tensor],
              target_grid: torch.Tensor,
              target_h: torch.Tensor,
              target_w: torch.Tensor,
              grid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute grid-level loss.

    Args:
        pred: output of GridPredictionHead.forward()
        target_grid: [batch, max_grid, max_grid] with values 0-9, padded
        target_h: [batch] true output heights
        target_w: [batch] true output widths
        grid_mask: [batch, max_grid, max_grid] bool, True for valid cells

    Returns:
        dict with 'total', 'cell_loss', 'dim_loss' tensors
    """
    grid_logits = pred['grid_logits']   # [batch, H, W, 10]
    dim_logits = pred['dim_logits']     # [batch, 2, max_grid]

    # Cell classification loss (only on valid cells)
    cell_logits_flat = grid_logits.reshape(-1, NUM_COLORS)
    target_flat = target_grid.reshape(-1).long()
    mask_flat = grid_mask.reshape(-1).float()

    cell_ce = F.cross_entropy(cell_logits_flat, target_flat, reduction='none')
    cell_loss = (cell_ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)

    # Dimension prediction loss
    h_loss = F.cross_entropy(dim_logits[:, 0, :], (target_h - 1).long().clamp(0))
    w_loss = F.cross_entropy(dim_logits[:, 1, :], (target_w - 1).long().clamp(0))
    dim_loss = (h_loss + w_loss) / 2

    total = cell_loss + 0.5 * dim_loss

    return {
        'total': total,
        'cell_loss': cell_loss,
        'dim_loss': dim_loss,
    }


def predict_grid(pred: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int, int]:
    """Extract predicted grid from model output.

    Returns:
        grid: [H, W] tensor with predicted colors
        h: predicted height
        w: predicted width
    """
    grid_logits = pred['grid_logits'][0]  # [max_grid, max_grid, 10]
    dim_logits = pred['dim_logits'][0]    # [2, max_grid]

    h = dim_logits[0].argmax().item() + 1
    w = dim_logits[1].argmax().item() + 1

    grid = grid_logits[:h, :w, :].argmax(dim=-1)  # [h, w]
    return grid, h, w
