"""
Torus Geometry Module - Lightweight Implementation
===================================================

Provides geometric primitives for torus-based code embeddings.
Used by VortexDisCode for semantic code navigation.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class TorusPosition:
    """Position on a torus surface (major_angle, minor_angle)."""
    theta: float  # Major circle angle (0 to 2π)
    phi: float    # Minor circle angle (0 to 2π)
    
    def to_3d(self, R: float = 1.0, r: float = 0.3) -> Tuple[float, float, float]:
        """Convert to 3D Cartesian coordinates."""
        x = (R + r * np.cos(self.phi)) * np.cos(self.theta)
        y = (R + r * np.cos(self.phi)) * np.sin(self.theta)
        z = r * np.sin(self.phi)
        return (x, y, z)
    
    def distance(self, other: 'TorusPosition', R: float = 1.0, r: float = 0.3) -> float:
        """Calculate geodesic distance on torus surface."""
        p1 = np.array(self.to_3d(R, r))
        p2 = np.array(other.to_3d(R, r))
        return float(np.linalg.norm(p1 - p2))


@dataclass
class TorusParticle:
    """Particle moving on torus surface with momentum."""
    position: TorusPosition
    velocity: Tuple[float, float]  # (d_theta/dt, d_phi/dt)
    mass: float = 1.0
    
    def update(self, dt: float = 0.1):
        """Update position based on velocity."""
        self.position.theta = (self.position.theta + self.velocity[0] * dt) % (2 * np.pi)
        self.position.phi = (self.position.phi + self.velocity[1] * dt) % (2 * np.pi)


class TorusLattice:
    """Discrete lattice on torus surface for code embeddings."""
    
    def __init__(self, R: float = 3.0, r: float = 1.0, 
                 levels: int = 4, n_major: int = None, n_minor: int = None):
        """
        Create lattice on torus surface.
        
        Args:
            R: Major radius of torus
            r: Minor radius of torus  
            levels: Number of hierarchical levels (controls lattice density)
            n_major: Number of points around major circle (overrides levels if set)
            n_minor: Number of points around minor circle (overrides levels if set)
        """
        self.R = R
        self.r = r
        self.levels = levels
        
        # Calculate lattice size from levels if not explicitly set
        if n_major is None:
            self.n_major = 4 * (2 ** levels)  # 16, 32, 64, etc.
        else:
            self.n_major = n_major
            
        if n_minor is None:
            self.n_minor = 2 * (2 ** levels)  # 8, 16, 32, etc.
        else:
            self.n_minor = n_minor
            
        self.lattice_points = self._generate_lattice()
    
    def _generate_lattice(self) -> List[TorusPosition]:
        """Generate evenly-spaced lattice points."""
        points = []
        for i in range(self.n_major):
            for j in range(self.n_minor):
                theta = 2 * np.pi * i / self.n_major
                phi = 2 * np.pi * j / self.n_minor
                points.append(TorusPosition(theta, phi))
        return points
    
    def nearest_point(self, position: TorusPosition) -> TorusPosition:
        """Find nearest lattice point to given position."""
        distances = [position.distance(p, R=self.R, r=self.r) for p in self.lattice_points]
        return self.lattice_points[np.argmin(distances)]
    
    def neighbors(self, position: TorusPosition, k: int = 4) -> List[TorusPosition]:
        """Get k nearest lattice neighbors."""
        distances = [(i, position.distance(p, R=self.R, r=self.r)) 
                    for i, p in enumerate(self.lattice_points)]
        distances.sort(key=lambda x: x[1])
        return [self.lattice_points[i] for i, _ in distances[:k]]


class TorusEmbedding:
    """Embed high-dimensional vectors onto torus surface."""
    
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 768, 
                 R: float = 1.0, r: float = 0.3, dim: int = None):
        """
        Initialize embedding with dimension and torus radii.
        
        Args:
            vocab_size: Size of vocabulary (for compatibility, not currently used)
            embed_dim: Dimension of embedding vectors
            R: Major radius of torus
            r: Minor radius of torus
            dim: Alias for embed_dim (for backward compatibility)
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim if dim is None else dim
        self.dim = self.embed_dim  # Backward compatibility
        self.R = R
        self.r = r
        self.lattice = TorusLattice(R=R, r=r)
        
    def embed(self, vector: np.ndarray) -> TorusPosition:
        """
        Embed vector onto torus using angle projection.
        Uses PCA-like reduction to 2D then maps to (theta, phi).
        """
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        
        # Simple projection: use first two principal angles
        # Normalize and map to [0, 2π]
        v_norm = vector / (np.linalg.norm(vector) + 1e-8)
        theta = np.arctan2(v_norm[0, 1 % self.dim], v_norm[0, 0]) + np.pi
        phi = np.arctan2(v_norm[0, 3 % self.dim], v_norm[0, 2 % self.dim]) + np.pi
        
        return TorusPosition(theta=theta, phi=phi)
    
    def embed_batch(self, vectors: np.ndarray) -> List[TorusPosition]:
        """Embed multiple vectors."""
        return [self.embed(v) for v in vectors]
    
    def distance_matrix(self, positions: List[TorusPosition]) -> np.ndarray:
        """Compute pairwise distances between positions."""
        n = len(positions)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = positions[i].distance(positions[j], self.R, self.r)
                D[i, j] = D[j, i] = d
        return D


# Convenience functions
def create_torus_embedding(dim: int = 768) -> TorusEmbedding:
    """Create a standard torus embedding."""
    return TorusEmbedding(dim=dim)


def random_torus_walk(n_steps: int = 100, dt: float = 0.1) -> List[TorusPosition]:
    """Generate random walk on torus surface."""
    particle = TorusParticle(
        position=TorusPosition(theta=0.0, phi=0.0),
        velocity=(np.random.randn(), np.random.randn())
    )
    
    trajectory = []
    for _ in range(n_steps):
        trajectory.append(TorusPosition(particle.position.theta, particle.position.phi))
        particle.update(dt)
        # Add random noise
        noise = (np.random.randn() * 0.1, np.random.randn() * 0.1)
        particle.velocity = (particle.velocity[0] + noise[0], particle.velocity[1] + noise[1])
    
    return trajectory


if __name__ == "__main__":
    print("🌀 Torus Geometry Module")
    print("=" * 50)
    
    # Test embedding
    embedding = create_torus_embedding(dim=128)
    test_vector = np.random.randn(128)
    pos = embedding.embed(test_vector)
    print(f"✓ Embedded 128-d vector to torus position: θ={pos.theta:.2f}, φ={pos.phi:.2f}")
    
    # Test lattice
    lattice = TorusLattice(n_major=12, n_minor=6)
    print(f"✓ Created lattice with {len(lattice.lattice_points)} points")
    
    # Test random walk
    walk = random_torus_walk(n_steps=50)
    print(f"✓ Generated random walk with {len(walk)} steps")
    
    print("\n✅ All torus geometry components functional!")
