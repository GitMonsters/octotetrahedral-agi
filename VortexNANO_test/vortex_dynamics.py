#!/usr/bin/env python3
"""
Vortex Dynamics for Semantic Code Navigation
===========================================

Implements differential flow dynamics on torus geometry for semantic code navigation.
Uses vortex centers as semantic attractors that evolve based on code relationships.

Key Components:
- VortexDynamics: Main dynamics engine with flow field computation
- SemanticFlowModel: Neural model for learning semantic relationships
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from torus_geometry import TorusPosition


class SemanticFlowModel(nn.Module):
    """
    Neural model for learning semantic flow patterns on torus geometry.
    Maps code embeddings to flow vectors that guide navigation.
    """
    
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Flow prediction network
        self.flow_net = nn.Sequential(
            nn.Linear(embed_dim + 2, hidden_dim),  # +2 for theta, phi
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output: (d_theta, d_phi)
        )
        
    def forward(self, positions: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute flow vectors for given positions and embeddings.
        
        Args:
            positions: (N, 2) tensor of (theta, phi) positions
            embeddings: (N, embed_dim) tensor of semantic embeddings
            
        Returns:
            (N, 2) tensor of flow vectors (d_theta, d_phi)
        """
        # Concatenate position and embedding
        combined = torch.cat([positions, embeddings], dim=1)
        
        # Predict flow
        flow = self.flow_net(combined)
        
        return flow


class VortexDynamics:
    """
    Vortex dynamics engine for semantic code navigation on torus geometry.
    
    Models semantic relationships as vortex attractors on the torus surface.
    Each vortex center represents a semantic concept or code pattern.
    """
    
    def __init__(self, R: float = 3.0, r: float = 1.0, 
                 n_vortices: int = 8, device: str = 'cpu'):
        """
        Initialize vortex dynamics system.
        
        Args:
            R: Major radius of torus
            r: Minor radius of torus  
            n_vortices: Number of vortex centers (semantic attractors)
            device: 'cpu' or 'cuda'
        """
        self.R = R
        self.r = r
        self.n_vortices = n_vortices
        self.device = device
        
        # Initialize vortex centers uniformly on torus
        # Distribute evenly around major and minor circles
        theta_centers = torch.linspace(0, 2*np.pi, n_vortices, 
                                      dtype=torch.float32, device=device)
        phi_centers = torch.linspace(0, 2*np.pi, n_vortices,
                                    dtype=torch.float32, device=device)
        
        # Stagger for better coverage
        self.vortex_centers = torch.stack([
            theta_centers,
            torch.roll(phi_centers, shifts=n_vortices//2)
        ], dim=1)  # (n_vortices, 2)
        
        # Vortex strengths (positive = source, negative = sink)
        # Alternate pattern for interesting flow
        self.vortex_strengths = torch.ones(n_vortices, device=device)
        self.vortex_strengths[1::2] *= -0.5  # Every other vortex is weaker sink
        
        # Semantic flow model (optional, for learning)
        self.flow_model = SemanticFlowModel()
        
    def compute_flow_field(self, theta: np.ndarray, phi: np.ndarray,
                          semantic_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute flow field at given positions on torus.
        
        Args:
            theta: (N,) array of theta coordinates
            phi: (N,) array of phi coordinates
            semantic_weights: Optional (N, n_vortices) weights for semantic influence
            
        Returns:
            (N, 2) array of flow vectors (d_theta, d_phi)
        """
        # Convert to tensors
        theta_t = torch.from_numpy(theta).float().to(self.device)
        phi_t = torch.from_numpy(phi).float().to(self.device)
        
        N = len(theta)
        flow = torch.zeros(N, 2, device=self.device)
        
        # Compute influence from each vortex
        for i in range(self.n_vortices):
            vortex_theta = self.vortex_centers[i, 0]
            vortex_phi = self.vortex_centers[i, 1]
            
            # Angular distance on torus (geodesic approximation)
            d_theta = theta_t - vortex_theta
            d_phi = phi_t - vortex_phi
            
            # Wrap to [-pi, pi]
            d_theta = torch.atan2(torch.sin(d_theta), torch.cos(d_theta))
            d_phi = torch.atan2(torch.sin(d_phi), torch.cos(d_phi))
            
            # Distance (approximate geodesic)
            dist = torch.sqrt(d_theta**2 + d_phi**2)
            
            # Avoid singularity at vortex center
            dist = torch.clamp(dist, min=0.1)
            
            # Flow strength (1/r for vortex)
            strength = self.vortex_strengths[i] / dist
            
            # Apply semantic weighting if provided
            if semantic_weights is not None:
                weights_t = torch.from_numpy(semantic_weights[:, i]).float().to(self.device)
                strength = strength * weights_t
            
            # Tangential flow (perpendicular to radial direction)
            # For vortex: flow is perpendicular to gradient
            flow[:, 0] += -strength * d_phi / dist  # d_theta component
            flow[:, 1] += strength * d_theta / dist  # d_phi component
        
        return flow.detach().cpu().numpy()
    
    def __call__(self, positions: List[Dict[str, Any]], 
                 time_step: float = 0.0) -> List[Dict[str, Any]]:
        """
        Evolve positions according to vortex dynamics.
        
        Args:
            positions: List of dicts with 'position' (TorusPosition), 
                      'activation', and 'particle' keys
            time_step: Time step for evolution (0 = no evolution, just flow)
            
        Returns:
            Evolved positions with same structure
        """
        if not positions:
            return []
        
        # Extract positions
        theta_list = []
        phi_list = []
        
        for pos_dict in positions:
            pos = pos_dict['position']
            theta_list.append(pos.theta)
            phi_list.append(pos.phi)
        
        theta = np.array(theta_list)
        phi = np.array(phi_list)
        
        # Compute flow field
        flow = self.compute_flow_field(theta, phi)
        
        # Evolve positions
        if time_step > 0:
            theta_new = theta + flow[:, 0] * time_step
            phi_new = phi + flow[:, 1] * time_step
            
            # Wrap to [0, 2*pi]
            theta_new = np.mod(theta_new, 2*np.pi)
            phi_new = np.mod(phi_new, 2*np.pi)
        else:
            theta_new = theta
            phi_new = phi
        
        # Create evolved positions
        evolved = []
        for i, pos_dict in enumerate(positions):
            new_pos = TorusPosition(theta=theta_new[i], phi=phi_new[i])
            evolved.append({
                'position': new_pos,
                'activation': pos_dict['activation'],
                'particle': pos_dict['particle'],
                'flow': flow[i]  # Add flow information
            })
        
        return evolved
    
    def update_vortex_centers(self, new_centers: np.ndarray):
        """
        Update vortex center positions (for learning/adaptation).
        
        Args:
            new_centers: (n_vortices, 2) array of (theta, phi) positions
        """
        self.vortex_centers = torch.from_numpy(new_centers).float().to(self.device)
    
    def update_vortex_strengths(self, new_strengths: np.ndarray):
        """
        Update vortex strengths (for learning/adaptation).
        
        Args:
            new_strengths: (n_vortices,) array of strength values
        """
        self.vortex_strengths = torch.from_numpy(new_strengths).float().to(self.device)


# Test if run directly
if __name__ == "__main__":
    print("🌀 Testing Vortex Dynamics...")
    
    # Create dynamics system
    dynamics = VortexDynamics(R=3.0, r=1.0, n_vortices=8, device='cpu')
    print(f"✓ Created {dynamics.n_vortices} vortex centers")
    
    # Test flow field computation
    theta = np.linspace(0, 2*np.pi, 20)
    phi = np.linspace(0, 2*np.pi, 20)
    flow = dynamics.compute_flow_field(theta, phi)
    print(f"✓ Computed flow field: {flow.shape}")
    print(f"  Mean flow magnitude: {np.linalg.norm(flow, axis=1).mean():.3f}")
    
    # Test position evolution
    test_positions = []
    for t, p in zip(theta[:5], phi[:5]):
        pos = TorusPosition(theta=t, phi=p)
        test_positions.append({
            'position': pos,
            'activation': 1.0,
            'particle': f'test_{len(test_positions)}'
        })
    
    evolved = dynamics(test_positions, time_step=0.1)
    print(f"✓ Evolved {len(evolved)} positions")
    for i, (orig, evol) in enumerate(zip(test_positions, evolved)):
        orig_pos = orig['position']
        evol_pos = evol['position']
        dist = orig_pos.distance(evol_pos, R=3.0, r=1.0)
        print(f"  Position {i}: moved {dist:.4f} units")
    
    # Test semantic flow model
    flow_model = SemanticFlowModel(embed_dim=512, hidden_dim=256)
    test_pos_tensor = torch.randn(10, 2)  # 10 positions
    test_embed_tensor = torch.randn(10, 512)  # 10 embeddings
    predicted_flow = flow_model(test_pos_tensor, test_embed_tensor)
    print(f"✓ Semantic flow model: {predicted_flow.shape}")
    
    print("\n✅ All vortex dynamics tests passed!")
