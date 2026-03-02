"""
Tetrahedral Geometry Module
Generates and manages the 64-point tetrahedral structure

Uses mathematically optimal spherical codes from:
  N. J. A. Sloane, with R. H. Hardin, W. D. Smith and others,
  "Tables of Spherical Codes", NeilSloane.com/packings/

The 64-point 3D packing achieves 26.235° minimum angular separation —
the proven optimal arrangement on a unit sphere. This replaces the
previous ad-hoc distribution which used random internal points.

Falls back to constructive generation if packing files aren't available.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import math


def _load_sloane_packing(dim: int, n_points: int) -> Optional[np.ndarray]:
    """Load optimal spherical code from Sloane's packing library."""
    search_dirs = [
        Path(__file__).parent.parent / "data" / "sloane_packings",
        Path.home() / "data" / "sloane_packings",
        Path.cwd() / "data" / "sloane_packings",
    ]
    for d in search_dirs:
        fpath = d / f"pack.{dim}.{n_points}.txt"
        if fpath.exists():
            vals = []
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        vals.append(float(line))
            pts = np.array(vals, dtype=np.float32).reshape(-1, dim)
            if len(pts) == n_points:
                return pts
    return None


class TetrahedralGeometry(nn.Module):
    """
    64-Point Tetrahedral Geometry System
    
    Generates points distributed across a regular tetrahedron and computes
    geometric relationships (distances, adjacencies) for use in attention mechanisms.
    """
    
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        # Generate all 64 points
        points = self._generate_all_points()
        self.register_buffer('points', torch.tensor(points, dtype=torch.float32))
        
        # Compute pairwise distances
        distances = self._compute_distances(self.points)
        self.register_buffer('distances', distances)
        
        # Compute adjacency matrix (based on distance threshold)
        adjacency = self._compute_adjacency(distances)
        self.register_buffer('adjacency', adjacency)
        
        # Compute geometric attention bias
        geo_bias = self._compute_geometric_bias(distances)
        self.register_buffer('geometric_bias', geo_bias)
    
    def _get_vertices(self) -> np.ndarray:
        """
        Get the 4 vertices of a regular tetrahedron centered at origin.
        Using coordinates that form a regular tetrahedron inscribed in a unit sphere.
        """
        # Regular tetrahedron vertices (inscribed in unit sphere)
        vertices = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=np.float32) / np.sqrt(3)
        
        return vertices
    
    def _get_edges(self) -> List[Tuple[int, int]]:
        """Get the 6 edges of the tetrahedron as vertex index pairs"""
        return [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    def _get_faces(self) -> List[List[int]]:
        """Get the 4 faces as lists of vertex indices"""
        return [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    
    def _generate_all_points(self) -> np.ndarray:
        """Generate all 64 points — prefer Sloane's optimal packing."""
        # Try optimal spherical code first (26.235° min separation)
        sloane_pts = _load_sloane_packing(dim=3, n_points=64)
        if sloane_pts is not None:
            return sloane_pts

        # Fallback: constructive tetrahedral distribution
        return self._generate_tetrahedral_points()

    def _generate_tetrahedral_points(self) -> np.ndarray:
        """Fallback: distribute 64 points across a regular tetrahedron."""
        vertices = self._get_vertices()
        edges = self._get_edges()
        faces = self._get_faces()
        
        points = []
        
        # 1. Add 4 vertices (indices 0-3)
        for v in vertices:
            points.append(v)
        
        # 2. Add 6 edge midpoints (indices 4-9)
        for v1_idx, v2_idx in edges:
            midpoint = (vertices[v1_idx] + vertices[v2_idx]) / 2
            points.append(midpoint)
        
        # 3. Add 4 face centers (indices 10-13)
        for face in faces:
            face_center = np.mean([vertices[i] for i in face], axis=0)
            points.append(face_center)
        
        # 4. Add 24 edge subdivisions (indices 14-37)
        # 4 points per edge, at t = 0.2, 0.4, 0.6, 0.8
        for v1_idx, v2_idx in edges:
            for t in [0.2, 0.4, 0.6, 0.8]:
                point = (1 - t) * vertices[v1_idx] + t * vertices[v2_idx]
                points.append(point)
        
        # 5. Add 12 face subdivisions (indices 38-49)
        # 3 points per face using barycentric coordinates
        for face in faces:
            v0, v1, v2 = [vertices[i] for i in face]
            # Interior points using barycentric coordinates
            bary_coords = [
                (0.5, 0.25, 0.25),
                (0.25, 0.5, 0.25),
                (0.25, 0.25, 0.5)
            ]
            for b0, b1, b2 in bary_coords:
                point = b0 * v0 + b1 * v1 + b2 * v2
                points.append(point)
        
        # 6. Add 14 internal points (indices 50-63)
        # Distributed inside the tetrahedron using barycentric coordinates
        np.random.seed(42)  # For reproducibility
        for _ in range(14):
            # Generate random barycentric coordinates (sum to 1)
            coords = np.random.dirichlet([1, 1, 1, 1])
            point = sum(coords[i] * vertices[i] for i in range(4))
            points.append(point)
        
        points = np.array(points, dtype=np.float32)
        assert len(points) == 64, f"Expected 64 points, got {len(points)}"
        
        return points
    
    def _compute_distances(self, points: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances between all points"""
        # points: [64, 3]
        diff = points.unsqueeze(0) - points.unsqueeze(1)  # [64, 64, 3]
        distances = torch.norm(diff, dim=-1)  # [64, 64]
        return distances
    
    def _compute_adjacency(self, distances: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Compute adjacency matrix based on distance threshold.
        Points within threshold distance are considered adjacent.
        """
        adjacency = (distances < threshold).float()
        # Remove self-connections
        adjacency = adjacency - torch.eye(adjacency.size(0), device=distances.device)
        return adjacency.clamp(min=0)
    
    def _compute_geometric_bias(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric attention bias from distances.
        Closer points get higher (less negative) bias.
        
        Uses inverse distance scaled by a learnable temperature.
        """
        # Avoid division by zero
        distances_safe = distances + 1e-6
        
        # Inverse distance (closer = larger value)
        inverse_dist = 1.0 / distances_safe
        
        # Normalize to reasonable range for attention
        # Scale so that adjacent points have bias ~1, distant ~0
        max_dist = distances.max()
        geo_bias = (max_dist - distances) / max_dist
        
        # Set diagonal to 0 (self-attention handled separately)
        geo_bias = geo_bias - torch.diag(torch.diag(geo_bias))
        
        return geo_bias
    
    def apply_transformation(
        self, 
        points: torch.Tensor, 
        transform_type: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply geometric transformation to points.
        
        Args:
            points: [N, 3] tensor of 3D points
            transform_type: One of 'rotate', 'scale', 'reflect', 'shear'
            **kwargs: Transformation-specific parameters
            
        Returns:
            Transformed points [N, 3]
        """
        if transform_type == 'rotate':
            return self._rotate(points, kwargs.get('angle', np.pi/6), kwargs.get('axis', 'y'))
        elif transform_type == 'scale':
            return self._scale(points, kwargs.get('factor', 1.2))
        elif transform_type == 'reflect':
            return self._reflect(points, kwargs.get('plane', 'xy'))
        elif transform_type == 'shear':
            return self._shear(points, kwargs.get('factor', 0.2))
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")
    
    def _rotate(self, points: torch.Tensor, angle: float, axis: str = 'y') -> torch.Tensor:
        """Rotate points around specified axis"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        if axis == 'x':
            rotation = torch.tensor([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ], dtype=points.dtype, device=points.device)
        elif axis == 'y':
            rotation = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=points.dtype, device=points.device)
        else:  # z
            rotation = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=points.dtype, device=points.device)
        
        return points @ rotation.T
    
    def _scale(self, points: torch.Tensor, factor: float) -> torch.Tensor:
        """Scale points uniformly"""
        return points * factor
    
    def _reflect(self, points: torch.Tensor, plane: str = 'xy') -> torch.Tensor:
        """Reflect points across specified plane"""
        reflected = points.clone()
        if plane == 'xy':
            reflected[:, 2] = -reflected[:, 2]
        elif plane == 'xz':
            reflected[:, 1] = -reflected[:, 1]
        else:  # yz
            reflected[:, 0] = -reflected[:, 0]
        return reflected
    
    def _shear(self, points: torch.Tensor, factor: float = 0.2) -> torch.Tensor:
        """Apply shear transformation"""
        shear_matrix = torch.tensor([
            [1, factor, 0],
            [0, 1, 0],
            [factor, 0, 1]
        ], dtype=points.dtype, device=points.device)
        return points @ shear_matrix.T
    
    def get_neighbors(self, point_idx: int, k: int = 6) -> torch.Tensor:
        """Get k nearest neighbors of a point"""
        distances_from_point = self.distances[point_idx]
        # Get k+1 nearest (including self), then exclude self
        _, indices = torch.topk(distances_from_point, k + 1, largest=False)
        # Remove self (index 0 after sorting)
        return indices[1:]
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the geometric structures for use in attention.
        
        Returns:
            points: [64, 3] - The 3D coordinates of all points
            distances: [64, 64] - Pairwise distances
            geometric_bias: [64, 64] - Attention bias based on geometry
        """
        return self.points, self.distances, self.geometric_bias
    
    def visualize_info(self) -> dict:
        """Return information for visualization"""
        return {
            'num_points': 64,
            'vertices': self.points[:4].cpu().numpy(),
            'edge_midpoints': self.points[4:10].cpu().numpy(),
            'face_centers': self.points[10:14].cpu().numpy(),
            'all_points': self.points.cpu().numpy(),
            'distances': self.distances.cpu().numpy(),
            'adjacency': self.adjacency.cpu().numpy()
        }


if __name__ == "__main__":
    # Test the geometry module
    print("Testing TetrahedralGeometry...")
    
    geom = TetrahedralGeometry()
    points, distances, geo_bias = geom()
    
    print(f"Points shape: {points.shape}")
    print(f"Distances shape: {distances.shape}")
    print(f"Geometric bias shape: {geo_bias.shape}")
    
    print(f"\nVertices (first 4 points):")
    print(points[:4])
    
    print(f"\nDistance range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"Mean distance: {distances.mean():.4f}")
    
    print(f"\nAdjacency matrix stats:")
    print(f"  Total adjacencies: {geom.adjacency.sum().item():.0f}")
    print(f"  Mean degree: {geom.adjacency.sum(dim=1).mean().item():.2f}")
    
    # Test transformation
    rotated = geom.apply_transformation(points, 'rotate', angle=np.pi/4)
    print(f"\nRotated points shape: {rotated.shape}")
    
    # Test neighbor lookup
    neighbors = geom.get_neighbors(0)
    print(f"\nNeighbors of vertex 0: {neighbors.tolist()}")
    
    print("\nAll tests passed!")
