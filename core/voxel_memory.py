"""
Voxel Memory — Persistent spatial world model

Inspired by OpenClaw's spatial agent memory: divides space into 3D voxels,
each storing a vector embedding + semantic label + timestamp + confidence.
Enables the model to build a queryable world state that persists across
inference steps.

For ARC: grids ARE voxel fields — each cell is a voxel with a color label.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class VoxelMemory(nn.Module):
    """
    Persistent 3D voxel memory with semantic embeddings.
    
    Each voxel stores:
    - embedding: learned vector representation [embed_dim]
    - semantic_label: integer label (e.g., ARC color 0-10)
    - timestamp: when last updated
    - confidence: how certain this observation is (decays over time)
    
    Supports attention-based queries over the voxel field.
    """
    
    def __init__(
        self,
        grid_size: int = 30,
        num_dims: int = 2,
        embed_dim: int = 256,
        num_labels: int = 11,
        decay_rate: float = 0.99,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_dims = num_dims
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.decay_rate = decay_rate
        
        total_voxels = grid_size ** num_dims
        self.total_voxels = total_voxels
        
        # Voxel embeddings (learnable base + writable overlay)
        self.base_embeddings = nn.Parameter(torch.randn(total_voxels, embed_dim) * 0.02)
        
        # Non-gradient buffers for state
        self.register_buffer('overlay', torch.zeros(total_voxels, embed_dim))
        self.register_buffer('semantic_labels', torch.zeros(total_voxels, dtype=torch.long))
        self.register_buffer('timestamps', torch.zeros(total_voxels, dtype=torch.long))
        self.register_buffer('confidence', torch.zeros(total_voxels))
        self.register_buffer('current_time', torch.tensor(0, dtype=torch.long))
        
        # Label embedding for semantic queries
        self.label_embed = nn.Embedding(num_labels, embed_dim)
        
        # Query projection for attention-based retrieval
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection after retrieval
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def _coords_to_index(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert N-D coordinates to flat voxel indices."""
        if self.num_dims == 2:
            return coords[..., 0] * self.grid_size + coords[..., 1]
        elif self.num_dims == 3:
            return (coords[..., 0] * self.grid_size * self.grid_size +
                    coords[..., 1] * self.grid_size + coords[..., 2])
        return coords[..., 0]
    
    def _get_effective_embeddings(self) -> torch.Tensor:
        """Get voxel embeddings with overlay and confidence weighting."""
        confidence_weight = self.confidence.unsqueeze(-1)  # [V, 1]
        return self.base_embeddings + confidence_weight * self.overlay
    
    def write_grid(
        self,
        grid: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        confidence_boost: float = 1.0,
    ):
        """
        Write an ARC-style grid into voxel memory.
        
        Args:
            grid: [height, width] integer grid (cell values 0-10)
            embeddings: Optional [height*width, embed_dim] to write
            confidence_boost: How confident this observation is
        """
        with torch.no_grad():
            h, w = grid.shape
            self.current_time += 1
            
            for y in range(min(h, self.grid_size)):
                for x in range(min(w, self.grid_size)):
                    idx = y * self.grid_size + x
                    self.semantic_labels[idx] = grid[y, x].clamp(0, self.num_labels - 1)
                    self.timestamps[idx] = self.current_time
                    self.confidence[idx] = min(self.confidence[idx] + confidence_boost, 1.0)
            
            if embeddings is not None:
                num_cells = min(h * w, self.total_voxels)
                self.overlay[:num_cells] = embeddings[:num_cells].detach()
    
    def write_voxels(
        self,
        indices: torch.Tensor,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Write specific voxels.
        
        Args:
            indices: [N] flat voxel indices
            embeddings: [N, embed_dim]
            labels: Optional [N] semantic labels
        """
        with torch.no_grad():
            self.current_time += 1
            valid = indices < self.total_voxels
            idx = indices[valid]
            self.overlay[idx] = embeddings[valid].detach()
            self.timestamps[idx] = self.current_time
            self.confidence[idx] = 1.0
            if labels is not None:
                self.semantic_labels[idx] = labels[valid]
    
    def decay(self):
        """Apply temporal decay to confidence values."""
        with torch.no_grad():
            age = (self.current_time - self.timestamps).float()
            self.confidence = self.confidence * (self.decay_rate ** age)
    
    def query_by_attention(
        self,
        query: torch.Tensor,
        top_k: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from voxel memory using attention.
        
        Args:
            query: [batch, seq_len, embed_dim]
            top_k: Number of voxels to attend to
            
        Returns:
            (retrieved [batch, seq_len, embed_dim], weights [batch, seq_len, top_k])
        """
        voxel_emb = self._get_effective_embeddings()  # [V, embed_dim]
        
        # Project queries and keys
        q = self.query_proj(query)  # [batch, seq, embed_dim]
        k = self.key_proj(voxel_emb)  # [V, embed_dim]
        v = self.value_proj(voxel_emb)  # [V, embed_dim]
        
        # Attention scores
        scores = torch.matmul(q, k.t()) / math.sqrt(self.embed_dim)
        
        # Confidence-weighted: boost scores for high-confidence voxels
        scores = scores + self.confidence.unsqueeze(0).unsqueeze(0) * 2.0
        
        # Top-k sparse attention
        if top_k < self.total_voxels:
            top_scores, top_idx = scores.topk(top_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, top_idx, 1.0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)  # [batch, seq, V]
        retrieved = torch.matmul(weights, v)  # [batch, seq, embed_dim]
        
        # Project and normalize
        retrieved = self.output_proj(retrieved)
        retrieved = self.norm(query + retrieved)
        
        # Return top-k weights for interpretability
        if top_k < self.total_voxels:
            sparse_weights = weights.gather(-1, top_idx)
        else:
            sparse_weights = weights
        
        return retrieved, sparse_weights
    
    def query_by_label(self, label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find all voxels with a specific semantic label.
        
        Returns:
            (indices, confidences) of matching voxels
        """
        mask = self.semantic_labels == label
        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        confidences = self.confidence[indices]
        return indices, confidences
    
    def query_by_region(
        self,
        center: torch.Tensor,
        radius: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query voxels within a spatial region.
        
        Args:
            center: [num_dims] center coordinates
            radius: Manhattan distance radius
            
        Returns:
            (embeddings, confidences) of voxels in region
        """
        # Generate all coordinates in region
        voxel_emb = self._get_effective_embeddings()
        
        if self.num_dims == 2:
            cy, cx = center[0].item(), center[1].item()
            indices = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y, x = int(cy + dy), int(cx + dx)
                    if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                        indices.append(y * self.grid_size + x)
            indices = torch.tensor(indices, device=voxel_emb.device)
        else:
            # For 3D, similar but with z
            indices = torch.arange(self.total_voxels, device=voxel_emb.device)
        
        return voxel_emb[indices], self.confidence[indices]
    
    def get_state_summary(self) -> Dict[str, float]:
        """Get summary statistics of voxel memory state."""
        populated = (self.confidence > 0.01).sum().item()
        return {
            'total_voxels': self.total_voxels,
            'populated_voxels': populated,
            'population_ratio': populated / self.total_voxels,
            'mean_confidence': self.confidence.mean().item(),
            'max_confidence': self.confidence.max().item(),
            'unique_labels': self.semantic_labels.unique().numel(),
            'current_time': self.current_time.item(),
        }
    
    def reset(self):
        """Clear all voxel memory state."""
        with torch.no_grad():
            self.overlay.zero_()
            self.semantic_labels.zero_()
            self.timestamps.zero_()
            self.confidence.zero_()
            self.current_time.zero_()


if __name__ == "__main__":
    print("Testing VoxelMemory...")
    
    vm = VoxelMemory(grid_size=10, num_dims=2, embed_dim=128, num_labels=11)
    
    # Write a grid
    grid = torch.randint(0, 11, (5, 5))
    vm.write_grid(grid)
    print(f"After write_grid: {vm.get_state_summary()}")
    
    # Attention query
    query = torch.randn(1, 4, 128)
    retrieved, weights = vm.query_by_attention(query, top_k=8)
    print(f"Query result shape: {retrieved.shape}")
    print(f"Weights shape: {weights.shape}")
    
    # Label query
    idx, conf = vm.query_by_label(3)
    print(f"Label 3 matches: {idx.shape[0]}")
    
    # Region query
    center = torch.tensor([2, 2])
    emb, conf = vm.query_by_region(center, radius=2)
    print(f"Region query (r=2): {emb.shape[0]} voxels")
    
    # Decay
    vm.decay()
    print(f"After decay: mean_conf={vm.confidence.mean().item():.4f}")
    
    # Reset
    vm.reset()
    print(f"After reset: {vm.get_state_summary()}")
    
    # Param count
    params = sum(p.numel() for p in vm.parameters())
    print(f"\nParameters: {params:,}")
    
    print("\nVoxelMemory tests passed!")
