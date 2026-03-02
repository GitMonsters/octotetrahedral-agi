"""
Buckminster Fuller's Synergetics & Tensegrity for OctoTetrahedral AGI

Implements Fuller's geometric principles:
1. Vector Equilibrium (Cuboctahedron) - the "zero-point" of energy
2. Tensegrity - tension/compression networks (islands of compression in a sea of tension)
3. Jitterbug Transformation - VE ↔ Octahedron ↔ Tetrahedron
4. Synergetic Geometry - 60-degree coordination vs 90-degree
5. Geodesic Frequency - subdivision patterns
6. Closest Packing of Spheres - 12 around 1

Neural Network Mappings:
- Vector Equilibrium → Hub state (balanced, zero-point)
- Tensegrity → Attention (tension=queries, compression=keys)
- Jitterbug → State transitions (phase changes)
- Geodesic frequency → Layer depth subdivision
- 12-around-1 → 8 limbs + 4 meta-cognitive states

"Unity is plural and at minimum two." - R. Buckminster Fuller
"Universe is the aggregate of all humanity's consciously apprehended experiences."

Usage:
    ve = VectorEquilibrium(hidden_dim=256)
    tensegrity = TensegrityNetwork(num_struts=8, num_tendons=24)
    output = ve.jitterbug_transform(input_state, phase=0.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


# =============================================================================
# SYNERGETIC CONSTANTS (Fuller's numerology from "Synergetics")
# =============================================================================

class SynergeticConstants:
    """
    Fuller's key numbers and relationships from Synergetics.
    
    These emerge from closest-packing of spheres and 60° coordination.
    """
    
    # Tetrahedron as unity (not cube)
    TETRA_VOLUME = 1.0  # Unity
    OCTA_VOLUME = 4.0   # 4 tetrahedra
    CUBE_VOLUME = 3.0   # 3 tetrahedra (not 1!)
    VE_VOLUME = 20.0    # Vector Equilibrium = 20 tetrahedra
    
    # Icosahedron (slightly less than VE)
    ICOSA_VOLUME = 18.51  # ~18.51 tetrahedra
    
    # Key angles in synergetic geometry
    TETRA_ANGLE = 60.0          # degrees - fundamental
    OCTA_ANGLE = 90.0           # degrees
    ICOSA_ANGLE = 72.0          # degrees (360/5)
    GOLDEN_ANGLE = 137.507764   # degrees (related to phi)
    
    # Golden ratio (appears in icosahedron)
    PHI = (1 + math.sqrt(5)) / 2  # 1.618...
    
    # Vector Equilibrium properties
    VE_VERTICES = 12     # 12 around 1
    VE_EDGES = 24        # 24 edges
    VE_FACES = 14        # 8 triangles + 6 squares
    VE_TRIANGLES = 8
    VE_SQUARES = 6
    
    # Jitterbug transformation phases
    JITTERBUG_VE = 0.0           # Vector Equilibrium
    JITTERBUG_ICOSA = 0.25       # Icosahedron (contracted)
    JITTERBUG_OCTA = 0.5         # Octahedron
    JITTERBUG_TETRA = 1.0        # Tetrahedron (fully contracted)
    
    # Frequency (geodesic subdivision)
    FREQ_1 = 1   # Basic polyhedron
    FREQ_2 = 2   # First subdivision (4x faces)
    FREQ_3 = 3   # Second subdivision (9x faces)
    FREQ_4 = 4   # Third subdivision (16x faces)
    
    # Tensegrity ratios (typical)
    TENSION_COMPRESSION_RATIO = PHI  # Golden ratio often appears
    
    # Synergetic coordinate system (quadray)
    QUADRAY_AXES = 4  # 4 axes from tetrahedron vertices


# =============================================================================
# VECTOR EQUILIBRIUM (CUBOCTAHEDRON)
# =============================================================================

class VectorEquilibrium(nn.Module):
    """
    Vector Equilibrium - Fuller's "zero-point" geometry.
    
    The VE is the only polyhedron where all vectors from center
    to vertices are equal AND all edge vectors are equal.
    
    Properties:
    - 12 vertices (12 around 1)
    - 24 edges (all equal length)
    - 14 faces (8 triangles + 6 squares)
    - Radial vectors = Edge vectors (unique property!)
    
    In neural terms:
    - VE represents perfect balance (equilibrium state)
    - Jitterbug transformation = state phase transitions
    - 12 vertices = 12 attention heads or feature channels
    """
    
    def __init__(self, hidden_dim: int, num_channels: int = 12):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels  # Maps to 12 vertices
        
        # VE vertex positions (normalized)
        # Vertices lie at midpoints of cube edges
        self.register_buffer('ve_vertices', self._compute_ve_vertices())
        
        # Learnable vertex activations
        self.vertex_weights = nn.Parameter(torch.ones(num_channels) / num_channels)
        
        # Jitterbug phase parameter (0=VE, 0.5=Octa, 1=Tetra)
        self.jitterbug_phase = nn.Parameter(torch.tensor(0.0))
        
        # Projections for each vertex channel
        self.vertex_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // num_channels)
            for _ in range(num_channels)
        ])
        
        # Reconstruction from vertices
        # 12 vertices * (hidden_dim // 12) = slightly less due to integer division
        self.vertex_output_dim = 12 * (hidden_dim // num_channels)
        self.reconstruct = nn.Linear(self.vertex_output_dim, hidden_dim)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
    
    def _compute_ve_vertices(self) -> torch.Tensor:
        """
        Compute the 12 vertices of Vector Equilibrium.
        
        VE vertices are at (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1)
        normalized to unit sphere.
        """
        vertices = []
        
        # Generate all permutations of (±1, ±1, 0)
        for coords in [(1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
                       (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
                       (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)]:
            vertices.append(coords)
        
        vertices = torch.tensor(vertices, dtype=torch.float32)
        
        # Normalize to unit sphere
        vertices = vertices / vertices.norm(dim=1, keepdim=True)
        
        return vertices
    
    def jitterbug_transform(self, vertices: torch.Tensor, phase: float) -> torch.Tensor:
        """
        Apply Jitterbug transformation.
        
        The Jitterbug is Fuller's discovery that VE can transform
        through icosahedron to octahedron to tetrahedron by 
        rotating triangular faces.
        
        phase=0: Vector Equilibrium (12 vertices equidistant)
        phase=0.5: Octahedron (6 vertices, pairs merged)
        phase=1: Tetrahedron (4 vertices, triplets merged)
        
        Args:
            vertices: [12, 3] vertex positions
            phase: transformation phase 0-1
            
        Returns:
            transformed: [12, 3] transformed positions
        """
        # Interpolate vertex positions based on phase
        # VE → Octa: vertices move toward octahedron positions
        # Octa → Tetra: vertices continue to tetrahedron positions
        
        if phase < 0.5:
            # VE → Octa
            t = phase * 2  # 0 to 1
            octa_vertices = self._compute_octa_from_ve(vertices)
            return vertices * (1 - t) + octa_vertices * t
        else:
            # Octa → Tetra
            t = (phase - 0.5) * 2  # 0 to 1
            octa_vertices = self._compute_octa_from_ve(vertices)
            tetra_vertices = self._compute_tetra_from_octa(octa_vertices)
            return octa_vertices * (1 - t) + tetra_vertices * t
    
    def _compute_octa_from_ve(self, ve_vertices: torch.Tensor) -> torch.Tensor:
        """VE vertices collapse to octahedron (pairs merge)"""
        # Octahedron has 6 vertices at (±1,0,0), (0,±1,0), (0,0,±1)
        octa_targets = torch.tensor([
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ], dtype=torch.float32, device=ve_vertices.device)
        return octa_targets
    
    def _compute_tetra_from_octa(self, octa_vertices: torch.Tensor) -> torch.Tensor:
        """Octahedron vertices collapse to tetrahedron (triplets merge)"""
        # Tetrahedron has 4 vertices
        r = 1.0 / math.sqrt(3)
        tetra_targets = torch.tensor([
            [r, r, r], [-r, -r, r], [-r, r, -r], [r, -r, -r],
            [r, r, r], [-r, -r, r], [-r, r, -r], [r, -r, -r],
            [r, r, r], [-r, -r, r], [-r, r, -r], [r, -r, -r]
        ], dtype=torch.float32, device=octa_vertices.device)
        return tetra_targets
    
    def compute_equilibrium_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Loss encouraging Vector Equilibrium state.
        
        In VE, all radial vectors are equal - this means
        all channel activations should be balanced.
        """
        # Activations should be uniform across channels
        mean_activation = activations.mean()
        variance = ((activations - mean_activation) ** 2).mean()
        
        return variance
    
    def forward(
        self, 
        x: torch.Tensor,
        return_vertices: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Vector Equilibrium geometry.
        
        Args:
            x: Input [batch, seq, hidden]
            return_vertices: Whether to return vertex activations
            
        Returns:
            output: [batch, seq, hidden]
            vertices: Optional [batch, seq, 12, hidden//12]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to each vertex channel
        vertex_outputs = []
        for i, proj in enumerate(self.vertex_projections):
            v_out = proj(x)  # [batch, seq, hidden//12]
            # Weight by vertex activation
            v_out = v_out * self.vertex_weights[i]
            vertex_outputs.append(v_out)
        
        # Stack vertices: [batch, seq, 12, hidden//12]
        vertices = torch.stack(vertex_outputs, dim=2)
        
        # Apply Jitterbug transformation to vertex weights
        phase = torch.sigmoid(self.jitterbug_phase).item()
        transformed_positions = self.jitterbug_transform(self.ve_vertices, phase)
        
        # Use transformed positions to modulate vertex mixing
        # Vertices closer together (in Jitterbug) should share more
        distances = torch.cdist(transformed_positions, transformed_positions)
        mixing_weights = F.softmax(-distances, dim=-1)  # [12, 12]
        
        # Mix vertex features based on Jitterbug geometry
        # vertices: [batch, seq, 12, dim]
        vertices_flat = vertices.reshape(batch_size * seq_len, 12, -1)
        mixed = torch.bmm(
            mixing_weights.unsqueeze(0).expand(batch_size * seq_len, -1, -1),
            vertices_flat
        )
        mixed = mixed.reshape(batch_size, seq_len, 12, -1)
        
        # Reconstruct to hidden dim
        output = mixed.reshape(batch_size, seq_len, -1)
        output = self.reconstruct(output)
        output = self.norm(output + x)  # Residual
        
        if return_vertices:
            return output, vertices
        return output, None


# =============================================================================
# TENSEGRITY NETWORKS
# =============================================================================

class TensegrityNetwork(nn.Module):
    """
    Tensegrity (Tensional Integrity) Network.
    
    Fuller's tensegrity principle: "Islands of compression in a sea of tension"
    
    Structure:
    - Struts (compression members) - don't touch each other
    - Tendons (tension members) - connect strut ends
    - System is pre-stressed (always under tension)
    
    Neural mapping:
    - Struts = Key vectors (compressed information)
    - Tendons = Query-Key attention (tension connections)
    - Pre-stress = Bias/baseline activation
    - Structural integrity = Attention coherence
    
    "The tensegrity icosahedron contains the most economical 
     relationship between compression and tension." - Fuller
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_struts: int = 6,  # 6-strut tensegrity is minimal
        tension_ratio: float = 1.618  # Golden ratio
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_struts = num_struts
        self.num_tendons = num_struts * 4  # Each strut end connects to 4 tendons
        self.tension_ratio = tension_ratio
        
        # Strut parameters (compression vectors)
        self.strut_directions = nn.Parameter(
            torch.randn(num_struts, hidden_dim) * 0.1
        )
        self.strut_lengths = nn.Parameter(torch.ones(num_struts))
        
        # Tendon parameters (tension connections)
        # Tendons connect strut endpoints
        self.tendon_stiffness = nn.Parameter(
            torch.ones(self.num_tendons) * tension_ratio
        )
        
        # Pre-stress (baseline tension in network)
        self.prestress = nn.Parameter(torch.tensor(0.1))
        
        # Projections
        self.compress_proj = nn.Linear(hidden_dim, hidden_dim)  # Struts
        self.tension_proj = nn.Linear(hidden_dim, hidden_dim)   # Tendons
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def compute_strut_endpoints(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute strut endpoint positions.
        
        Each strut has two ends; in tensegrity they don't touch other struts.
        
        Returns:
            start_points: [num_struts, hidden_dim]
            end_points: [num_struts, hidden_dim]
        """
        # Normalize directions
        directions = F.normalize(self.strut_directions, dim=-1)
        
        # Strut endpoints are center ± (length/2 * direction)
        half_lengths = self.strut_lengths.unsqueeze(-1) / 2
        
        start_points = -half_lengths * directions
        end_points = half_lengths * directions
        
        return start_points, end_points
    
    def compute_tension_matrix(self) -> torch.Tensor:
        """
        Compute tension connections between strut endpoints.
        
        Returns:
            tension: [2*num_struts, 2*num_struts] connection matrix
        """
        start, end = self.compute_strut_endpoints()
        
        # All endpoints: [2*num_struts, hidden_dim]
        all_endpoints = torch.cat([start, end], dim=0)
        
        # Compute pairwise distances
        distances = torch.cdist(all_endpoints, all_endpoints)
        
        # Tension inversely proportional to distance (Hooke's law)
        # But strut endpoints on same strut shouldn't connect
        tension = 1.0 / (distances + 1e-6)
        
        # Mask out same-strut connections
        n = self.num_struts
        for i in range(n):
            tension[i, i + n] = 0  # start[i] to end[i]
            tension[i + n, i] = 0  # end[i] to start[i]
        
        # Apply prestress
        tension = tension + self.prestress
        
        # Normalize rows
        tension = F.softmax(tension, dim=-1)
        
        return tension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through tensegrity network.
        
        Compression (struts) and tension (tendons) work together
        to maintain structural integrity of the representation.
        
        Args:
            x: Input [batch, seq, hidden]
            
        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compression pathway (struts)
        compressed = self.compress_proj(x)
        
        # Project onto strut directions
        # strut_directions: [num_struts, hidden]
        # compressed: [batch, seq, hidden]
        strut_activations = torch.einsum(
            'bsh,nh->bsn',
            compressed,
            F.normalize(self.strut_directions, dim=-1)
        )  # [batch, seq, num_struts]
        
        # Scale by strut lengths
        strut_activations = strut_activations * self.strut_lengths
        
        # Tension pathway (tendons)
        tensioned = self.tension_proj(x)
        
        # Apply tension matrix (connects endpoints)
        tension_matrix = self.compute_tension_matrix()  # [2n, 2n]
        
        # Duplicate strut activations for endpoints
        endpoint_activations = torch.cat([strut_activations, strut_activations], dim=-1)
        
        # Apply tension connections
        tensioned_activations = torch.einsum(
            'bse,ef->bsf',
            endpoint_activations,
            tension_matrix
        )
        
        # Combine compression and tension
        # The system achieves integrity through their balance
        # Strut activations: [batch, seq, num_struts]
        # Use struts to modulate tensioned output
        strut_gate = torch.sigmoid(strut_activations.mean(dim=-1, keepdim=True))
        
        # Project strut influence back to hidden dim
        strut_proj = torch.einsum(
            'bsn,nh->bsh',
            strut_activations,
            F.normalize(self.strut_directions, dim=-1)
        )
        
        output = self.output_proj(strut_gate * strut_proj + (1 - strut_gate) * tensioned)
        output = self.norm(output + x)
        
        return output
    
    def get_structural_integrity(self) -> float:
        """
        Compute structural integrity score.
        
        High integrity = balanced tension throughout network.
        """
        tension_matrix = self.compute_tension_matrix()
        
        # Entropy of tension distribution (higher = more balanced)
        entropy = -(tension_matrix * torch.log(tension_matrix + 1e-10)).sum()
        max_entropy = math.log(tension_matrix.numel())
        
        return (entropy / max_entropy).item()


# =============================================================================
# GEODESIC FREQUENCY PATTERNS
# =============================================================================

class GeodesicFrequency(nn.Module):
    """
    Geodesic Frequency subdivision patterns.
    
    Fuller's geodesic domes use frequency to subdivide icosahedron faces.
    Frequency N means each edge is divided into N parts.
    
    Properties:
    - Frequency 1: Basic icosahedron (20 faces)
    - Frequency 2: 80 faces (each triangle → 4 triangles)
    - Frequency N: 20 * N² faces
    
    Neural mapping:
    - Frequency = Network depth/resolution
    - Higher frequency = finer detail processing
    - Geodesic = optimal structural distribution
    
    "Nature always uses the most economical structural strategy." - Fuller
    """
    
    def __init__(
        self,
        hidden_dim: int,
        base_frequency: int = 2,
        max_frequency: int = 4
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.base_frequency = base_frequency
        self.max_frequency = max_frequency
        
        # Learnable frequency (can adapt during training)
        self.frequency = nn.Parameter(
            torch.tensor(float(base_frequency))
        )
        
        # Subdivision projections for each frequency level
        self.freq_projections = nn.ModuleDict({
            str(f): nn.Linear(hidden_dim, hidden_dim)
            for f in range(1, max_frequency + 1)
        })
        
        # Great circle projections (geodesic paths)
        num_great_circles = 6  # Icosahedron has 6 primary great circles
        self.great_circle_weights = nn.Parameter(
            torch.ones(num_great_circles) / num_great_circles
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def compute_subdivision_weights(self) -> torch.Tensor:
        """
        Compute interpolation weights between frequency levels.
        
        Non-integer frequencies interpolate between adjacent levels.
        """
        freq = torch.clamp(self.frequency.float(), 1.0, float(self.max_frequency))
        floor_freq = int(torch.floor(freq).clamp(1, self.max_frequency).item())
        ceil_freq = min(floor_freq + 1, self.max_frequency)
        
        # Interpolation weight
        t = freq - floor_freq
        
        weights = torch.zeros(self.max_frequency)
        weights[floor_freq - 1] = 1 - t
        if ceil_freq <= self.max_frequency:
            weights[ceil_freq - 1] = t
        
        return weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply geodesic frequency processing.
        
        Higher frequencies process finer details, lower frequencies
        process broader patterns - like multi-scale processing.
        
        Args:
            x: Input [batch, seq, hidden]
            
        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_len, _ = x.shape
        
        # Get subdivision weights
        weights = self.compute_subdivision_weights().to(x.device)
        
        # Process at each frequency level
        freq_outputs = []
        for f in range(1, self.max_frequency + 1):
            proj = self.freq_projections[str(f)]
            
            # Number of geodesic "faces" at this frequency
            num_faces = 20 * f * f  # Icosahedron subdivision
            
            # Process with frequency-specific projection
            out = proj(x)
            
            # Apply frequency-dependent pooling
            # Higher frequency = less pooling (more detail)
            pool_size = max(1, seq_len // (f * f))
            if pool_size < seq_len:
                out = F.avg_pool1d(
                    out.transpose(1, 2),
                    kernel_size=pool_size,
                    stride=1,
                    padding=pool_size // 2
                ).transpose(1, 2)
                out = out[:, :seq_len, :]  # Trim to original length
            
            freq_outputs.append(out * weights[f - 1])
        
        # Combine frequency levels
        combined = sum(freq_outputs)
        
        # Apply great circle weighting
        # (great circles are the structural "ribs" of geodesic dome)
        gc_weighted = combined * self.great_circle_weights.mean()
        
        output = self.output_proj(gc_weighted)
        output = self.norm(output + x)
        
        return output


# =============================================================================
# CLOSEST PACKING OF SPHERES (12 AROUND 1)
# =============================================================================

class TwelveAroundOne(nn.Module):
    """
    Closest Packing of Spheres - 12 Around 1.
    
    Fuller's discovery that exactly 12 spheres of equal size
    can touch a central sphere (and each other optimally).
    
    This is the basis for:
    - Vector Equilibrium (12 vertices)
    - Atomic structure
    - Nuclear geometry
    
    Neural mapping:
    - Central sphere = Hub representation
    - 12 surrounding = Peripheral feature channels
    - Touching = Information exchange
    
    "The closest-packed spheres nest in the interstices
     of the previous layer." - Fuller
    """
    
    def __init__(self, hidden_dim: int, central_dim: Optional[int] = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.central_dim = central_dim or hidden_dim
        self.num_peripheral = 12
        
        # Central sphere processing
        self.central_proj = nn.Linear(hidden_dim, self.central_dim)
        
        # 12 peripheral sphere projections
        self.peripheral_projs = nn.ModuleList([
            nn.Linear(hidden_dim, self.central_dim // 4)
            for _ in range(self.num_peripheral)
        ])
        
        # Contact points between spheres
        # Central touches all 12, each peripheral touches 5 others
        self.contact_weights = nn.Parameter(
            self._init_contact_weights()
        )
        
        # Output reconstruction
        self.output_proj = nn.Linear(
            self.central_dim + 12 * (self.central_dim // 4),
            hidden_dim
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def _init_contact_weights(self) -> torch.Tensor:
        """
        Initialize contact weights based on sphere packing geometry.
        
        In 12-around-1 packing:
        - Central sphere contacts all 12
        - Each peripheral contacts central + 5 neighbors
        """
        # Adjacency for 12-around-1 (VE vertices + edges)
        # This is the cuboctahedron contact pattern
        weights = torch.zeros(13, 13)  # 1 central + 12 peripheral
        
        # Central (0) contacts all peripheral (1-12)
        weights[0, 1:] = 1.0
        weights[1:, 0] = 1.0
        
        # Peripheral contacts (VE edge structure)
        # Each vertex of VE connects to 4 neighbors
        ve_edges = [
            (1,2), (1,3), (1,5), (1,6),
            (2,1), (2,4), (2,6), (2,7),
            (3,1), (3,4), (3,5), (3,8),
            (4,2), (4,3), (4,7), (4,8),
            (5,1), (5,3), (5,9), (5,10),
            (6,1), (6,2), (6,9), (6,11),
            (7,2), (7,4), (7,11), (7,12),
            (8,3), (8,4), (8,10), (8,12),
            (9,5), (9,6), (9,10), (9,11),
            (10,5), (10,8), (10,9), (10,12),
            (11,6), (11,7), (11,9), (11,12),
            (12,7), (12,8), (12,10), (12,11)
        ]
        
        for i, j in ve_edges:
            weights[i, j] = 1.0
        
        # Normalize
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        
        return weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through 12-around-1 sphere packing.
        
        Args:
            x: Input [batch, seq, hidden]
            
        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_len, _ = x.shape
        
        # Central sphere processing
        central = self.central_proj(x)  # [batch, seq, central_dim]
        
        # Peripheral sphere processing
        peripherals = []
        for proj in self.peripheral_projs:
            p = proj(x)  # [batch, seq, central_dim//4]
            peripherals.append(p)
        
        # Stack: [batch, seq, 12, central_dim//4]
        peripheral_stack = torch.stack(peripherals, dim=2)
        
        # Apply contact-based mixing
        # Treat central specially
        central_broadcast = central.unsqueeze(2)  # [batch, seq, 1, central_dim]
        
        # Central influences peripherals
        central_to_peripheral = self.contact_weights[0, 1:].view(1, 1, 12, 1)
        peripheral_stack = peripheral_stack + central_broadcast[..., :self.central_dim//4] * central_to_peripheral * 0.1
        
        # Peripherals influence each other
        peripheral_contacts = self.contact_weights[1:, 1:]  # [12, 12]
        peripheral_mixed = torch.einsum(
            'bspd,pq->bsqd',
            peripheral_stack,
            peripheral_contacts
        )
        
        # Combine
        peripheral_flat = peripheral_mixed.reshape(batch_size, seq_len, -1)
        combined = torch.cat([central, peripheral_flat], dim=-1)
        
        output = self.output_proj(combined)
        output = self.norm(output + x)
        
        return output


# =============================================================================
# SYNERGETICS COORDINATE SYSTEM (QUADRAY)
# =============================================================================

class QuadrayCoordinates(nn.Module):
    """
    Quadray (Synergetics) Coordinate System.
    
    Fuller proposed replacing XYZ coordinates with 4 axes
    pointing from tetrahedron center to vertices.
    
    Properties:
    - 4 axes at 109.47° angles (tetrahedral)
    - All positive coordinates (no negatives needed)
    - More symmetric than XYZ
    - Natural for closest-packing description
    
    Conversion:
    - Quadray (a,b,c,d) where a+b+c+d = constant
    - XYZ can be recovered via transformation
    
    Neural mapping:
    - 4 quadray axes = 4 basis feature directions
    - Tetrahedral symmetry = balanced representation
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_axes = 4
        
        # Quadray basis vectors (tetrahedron vertices)
        # Pointing from center to vertices of regular tetrahedron
        r = 1.0 / math.sqrt(3)
        quadray_basis = torch.tensor([
            [r, r, r],      # Vertex 1
            [-r, -r, r],    # Vertex 2
            [-r, r, -r],    # Vertex 3
            [r, -r, -r]     # Vertex 4
        ], dtype=torch.float32)
        
        self.register_buffer('quadray_basis', quadray_basis)
        
        # Learnable quadray weights
        self.quadray_weights = nn.Parameter(torch.ones(4) / 4)
        
        # Projections
        self.xyz_to_quadray = nn.Linear(hidden_dim, 4 * (hidden_dim // 4))
        self.quadray_to_xyz = nn.Linear(4 * (hidden_dim // 4), hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def to_quadray(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Convert XYZ-like representation to quadray.
        
        In true quadray: q = M @ xyz + offset to make all positive
        """
        # Project to 4 quadray channels
        quadray = self.xyz_to_quadray(xyz)  # [batch, seq, 4*dim]
        
        # Reshape to [batch, seq, 4, dim//4]
        batch_size, seq_len = xyz.shape[:2]
        quadray = quadray.reshape(batch_size, seq_len, 4, -1)
        
        # Apply ReLU to ensure non-negative (quadray property)
        quadray = F.relu(quadray)
        
        # Weight by quadray axes
        weighted = quadray * self.quadray_weights.view(1, 1, 4, 1)
        
        return weighted
    
    def from_quadray(self, quadray: torch.Tensor) -> torch.Tensor:
        """
        Convert quadray back to XYZ-like representation.
        """
        # Flatten quadray channels
        flat = quadray.reshape(quadray.shape[0], quadray.shape[1], -1)
        
        # Project back
        xyz = self.quadray_to_xyz(flat)
        
        return xyz
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through quadray coordinate transformation.
        
        The round-trip through quadray imposes tetrahedral
        symmetry on the representation.
        """
        # Convert to quadray
        quadray = self.to_quadray(x)
        
        # Process in quadray space
        # Normalize sum (quadray constraint: sum is constant)
        quadray_sum = quadray.sum(dim=2, keepdim=True)
        quadray_normalized = quadray / (quadray_sum + 1e-6) * 4
        
        # Convert back
        output = self.from_quadray(quadray_normalized)
        
        return self.norm(output + x)


# =============================================================================
# UNIFIED FULLER GEOMETRY MODULE
# =============================================================================

class FullerSynergetics(nn.Module):
    """
    Unified Buckminster Fuller Synergetics Module.
    
    Combines all Fuller geometric principles:
    - Vector Equilibrium (zero-point balance)
    - Tensegrity (tension-compression integrity)
    - Geodesic Frequency (multi-scale subdivision)
    - 12-Around-1 (closest packing)
    - Quadray Coordinates (tetrahedral basis)
    
    "You never change things by fighting the existing reality.
     To change something, build a new model that makes the
     existing model obsolete." - R. Buckminster Fuller
    """
    
    def __init__(
        self,
        hidden_dim: int,
        enable_ve: bool = True,
        enable_tensegrity: bool = True,
        enable_geodesic: bool = True,
        enable_packing: bool = True,
        enable_quadray: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Sub-modules
        self.ve = VectorEquilibrium(hidden_dim) if enable_ve else None
        self.tensegrity = TensegrityNetwork(hidden_dim) if enable_tensegrity else None
        self.geodesic = GeodesicFrequency(hidden_dim) if enable_geodesic else None
        self.packing = TwelveAroundOne(hidden_dim) if enable_packing else None
        self.quadray = QuadrayCoordinates(hidden_dim) if enable_quadray else None
        
        # Combination weights (learnable)
        num_active = sum([
            enable_ve, enable_tensegrity, enable_geodesic,
            enable_packing, enable_quadray
        ])
        self.combination_weights = nn.Parameter(
            torch.ones(num_active) / num_active
        )
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process through all Fuller geometric principles.
        
        Returns dict with individual outputs for analysis.
        """
        outputs = {}
        weighted_outputs = []
        weight_idx = 0
        
        # Vector Equilibrium
        if self.ve is not None:
            ve_out, _ = self.ve(x)
            outputs['vector_equilibrium'] = ve_out
            weighted_outputs.append(ve_out * self.combination_weights[weight_idx])
            weight_idx += 1
        
        # Tensegrity
        if self.tensegrity is not None:
            tens_out = self.tensegrity(x)
            outputs['tensegrity'] = tens_out
            weighted_outputs.append(tens_out * self.combination_weights[weight_idx])
            weight_idx += 1
        
        # Geodesic Frequency
        if self.geodesic is not None:
            geo_out = self.geodesic(x)
            outputs['geodesic'] = geo_out
            weighted_outputs.append(geo_out * self.combination_weights[weight_idx])
            weight_idx += 1
        
        # 12-Around-1 Packing
        if self.packing is not None:
            pack_out = self.packing(x)
            outputs['packing_12'] = pack_out
            weighted_outputs.append(pack_out * self.combination_weights[weight_idx])
            weight_idx += 1
        
        # Quadray Coordinates
        if self.quadray is not None:
            quad_out = self.quadray(x)
            outputs['quadray'] = quad_out
            weighted_outputs.append(quad_out * self.combination_weights[weight_idx])
            weight_idx += 1
        
        # Combine all geometric outputs
        combined = sum(weighted_outputs)
        output = self.output_proj(combined)
        output = self.norm(output + x)
        
        outputs['combined'] = output
        
        return outputs
    
    def get_geometric_stats(self) -> Dict[str, float]:
        """Get statistics about the geometric processing."""
        stats = {}
        
        if self.ve is not None:
            stats['ve_jitterbug_phase'] = torch.sigmoid(self.ve.jitterbug_phase).item()
            stats['ve_equilibrium'] = self.ve.vertex_weights.std().item()
        
        if self.tensegrity is not None:
            stats['tensegrity_integrity'] = self.tensegrity.get_structural_integrity()
            stats['tensegrity_prestress'] = self.tensegrity.prestress.item()
        
        if self.geodesic is not None:
            stats['geodesic_frequency'] = self.geodesic.frequency.item()
        
        stats['combination_weights'] = self.combination_weights.softmax(dim=0).tolist()
        
        return stats


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Buckminster Fuller Synergetics Module...")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 32
    hidden_dim = 256
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test Vector Equilibrium
    print("\n1. Vector Equilibrium (Cuboctahedron):")
    ve = VectorEquilibrium(hidden_dim)
    ve_out, vertices = ve(x, return_vertices=True)
    print(f"   Input: {x.shape}")
    print(f"   Output: {ve_out.shape}")
    print(f"   Vertices: {vertices.shape}")
    print(f"   Jitterbug phase: {torch.sigmoid(ve.jitterbug_phase).item():.3f}")
    
    # Test Tensegrity
    print("\n2. Tensegrity Network:")
    tens = TensegrityNetwork(hidden_dim)
    tens_out = tens(x)
    print(f"   Output: {tens_out.shape}")
    print(f"   Structural integrity: {tens.get_structural_integrity():.3f}")
    print(f"   Pre-stress: {tens.prestress.item():.3f}")
    
    # Test Geodesic Frequency
    print("\n3. Geodesic Frequency:")
    geo = GeodesicFrequency(hidden_dim)
    geo_out = geo(x)
    print(f"   Output: {geo_out.shape}")
    print(f"   Current frequency: {geo.frequency.item():.2f}")
    
    # Test 12-Around-1
    print("\n4. Closest Packing (12 Around 1):")
    pack = TwelveAroundOne(hidden_dim)
    pack_out = pack(x)
    print(f"   Output: {pack_out.shape}")
    
    # Test Quadray
    print("\n5. Quadray Coordinates:")
    quad = QuadrayCoordinates(hidden_dim)
    quad_out = quad(x)
    print(f"   Output: {quad_out.shape}")
    print(f"   Quadray weights: {quad.quadray_weights.data.numpy().round(3)}")
    
    # Test Unified Module
    print("\n6. Unified Fuller Synergetics:")
    fuller = FullerSynergetics(hidden_dim)
    outputs = fuller(x)
    print(f"   Combined output: {outputs['combined'].shape}")
    print(f"   Stats: {fuller.get_geometric_stats()}")
    
    print("\n" + "=" * 60)
    print("All Fuller Synergetics tests passed!")
    print("\nKey Concepts Implemented:")
    print("- Vector Equilibrium (12 vertices, jitterbug transformation)")
    print("- Tensegrity (compression struts + tension tendons)")
    print("- Geodesic Frequency (icosahedral subdivision)")
    print("- 12-Around-1 Sphere Packing (closest packing)")
    print("- Quadray Coordinates (tetrahedral basis system)")
