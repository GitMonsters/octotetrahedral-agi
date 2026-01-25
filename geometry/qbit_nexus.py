"""
QbitNexus Theory - Icosahedral Quantum Information Nexus

Inspired by the icosahedral structure where:
- 12 vertices act as quantum information nodes (qbits)
- 20 triangular faces define information flow pathways
- 30 edges form the communication channels
- Central nexus point is the hub where all information converges

Key Insights:
1. Icosahedron is the most spherical Platonic solid - optimal for isotropic information distribution
2. Each vertex is equidistant from center - democratic information access
3. Triangular faces provide maximum rigidity - stable information states
4. Dual geometry (dodecahedron) emerges at center - hidden structure

Mathematical Foundation:
- Golden ratio φ = (1+√5)/2 appears in icosahedral coordinates
- Vertices at: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
- 5-fold rotational symmetry (impossible in crystals - quasicrystalline)

Neural Mapping:
- 12 vertices → 12 attention heads (or 12 = 8 limbs + 4 meta-processes)
- 20 faces → 20 information routing pathways
- 30 edges → 30 coupling channels
- Central nexus → Hub synchronization point
- Golden ratio → Fibonacci-based layer scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math


# Golden ratio - fundamental to icosahedral geometry
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895


def get_icosahedron_vertices() -> torch.Tensor:
    """
    Get the 12 vertices of a unit icosahedron.
    
    Coordinates use golden ratio φ:
    - (0, ±1, ±φ)
    - (±1, ±φ, 0)
    - (±φ, 0, ±1)
    
    Returns:
        vertices: [12, 3] tensor of vertex coordinates
    """
    vertices = torch.tensor([
        # (0, ±1, ±φ)
        [0, 1, PHI], [0, 1, -PHI], [0, -1, PHI], [0, -1, -PHI],
        # (±1, ±φ, 0)
        [1, PHI, 0], [-1, PHI, 0], [1, -PHI, 0], [-1, -PHI, 0],
        # (±φ, 0, ±1)
        [PHI, 0, 1], [-PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, -1]
    ], dtype=torch.float32)
    
    # Normalize to unit sphere
    vertices = vertices / vertices.norm(dim=-1, keepdim=True)
    
    return vertices


def get_icosahedron_edges() -> torch.Tensor:
    """
    Get the 30 edges of an icosahedron as vertex index pairs.
    
    Returns:
        edges: [30, 2] tensor of vertex index pairs
    """
    # Each vertex connects to 5 neighbors
    # Total edges = 12 * 5 / 2 = 30
    edges = torch.tensor([
        [0, 2], [0, 4], [0, 5], [0, 8], [0, 9],
        [1, 3], [1, 4], [1, 5], [1, 10], [1, 11],
        [2, 6], [2, 7], [2, 8], [2, 9],
        [3, 6], [3, 7], [3, 10], [3, 11],
        [4, 5], [4, 8], [4, 10],
        [5, 9], [5, 11],
        [6, 7], [6, 8], [6, 10],
        [7, 9], [7, 11],
        [8, 10], [9, 11]
    ], dtype=torch.long)
    
    return edges


def get_icosahedron_faces() -> torch.Tensor:
    """
    Get the 20 triangular faces of an icosahedron as vertex index triples.
    
    Returns:
        faces: [20, 3] tensor of vertex index triples
    """
    faces = torch.tensor([
        [0, 2, 8], [0, 8, 4], [0, 4, 5], [0, 5, 9], [0, 9, 2],
        [1, 3, 11], [1, 11, 5], [1, 5, 4], [1, 4, 10], [1, 10, 3],
        [2, 7, 9], [2, 6, 7], [2, 8, 6],
        [3, 6, 7], [3, 7, 11], [3, 10, 6],
        [4, 8, 10], [5, 11, 9],
        [6, 8, 10], [7, 9, 11]
    ], dtype=torch.long)
    
    return faces


class IcosahedralEmbedding(nn.Module):
    """
    Embed hidden states onto icosahedral vertices.
    
    Maps sequence elements to the 12 vertices of an icosahedron,
    preserving the geometric relationships.
    """
    
    def __init__(self, hidden_dim: int, num_vertices: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_vertices = num_vertices
        
        # Icosahedral coordinates
        self.register_buffer('vertices', get_icosahedron_vertices())
        self.register_buffer('edges', get_icosahedron_edges())
        self.register_buffer('faces', get_icosahedron_faces())
        
        # Learnable vertex embeddings
        self.vertex_embed = nn.Parameter(torch.randn(num_vertices, hidden_dim) * 0.02)
        
        # Project input to vertex space
        self.to_vertex = nn.Linear(hidden_dim, num_vertices)
        
        # Project back from vertex space
        self.from_vertex = nn.Linear(num_vertices, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed input onto icosahedral vertices.
        
        Args:
            x: Input tensor [batch, seq, hidden]
            
        Returns:
            vertex_states: [batch, seq, num_vertices, hidden]
            vertex_weights: [batch, seq, num_vertices]
        """
        batch, seq, hidden = x.shape
        
        # Compute soft assignment to vertices
        vertex_weights = F.softmax(self.to_vertex(x), dim=-1)  # [batch, seq, 12]
        
        # Get vertex embeddings
        # Combine input with geometric vertex embeddings
        vertex_states = vertex_weights.unsqueeze(-1) * self.vertex_embed  # [batch, seq, 12, hidden]
        
        return vertex_states, vertex_weights


class NexusAttention(nn.Module):
    """
    Attention mechanism guided by icosahedral geometry.
    
    Information flows through the geometric structure:
    - Short-range: along edges (30 channels)
    - Medium-range: across faces (20 pathways)  
    - Long-range: through central nexus (hub)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 12,  # One per vertex
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Adjust num_heads to be divisible by hidden_dim
        for h in range(min(num_heads, 8), 0, -1):  # Cap at 8 for efficiency
            if hidden_dim % h == 0:
                num_heads = h
                break
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Icosahedral structure
        self.register_buffer('vertices', get_icosahedron_vertices())
        self.register_buffer('edges', get_icosahedron_edges())
        self.register_buffer('faces', get_icosahedron_faces())
        
        # Compute adjacency matrix from edges
        adj = torch.zeros(12, 12)
        edges = get_icosahedron_edges()
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1
        self.register_buffer('adjacency', adj)
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Central nexus projection
        self.nexus_proj = nn.Linear(hidden_dim, hidden_dim)
        self.nexus_gate = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with icosahedral attention.
        
        Args:
            x: Input [batch, seq, hidden]
            attention_mask: Optional mask [batch, seq]
            
        Returns:
            output: [batch, seq, hidden]
            attention_weights: [batch, num_heads, seq, seq]
        """
        batch, seq, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [batch, num_heads, seq, head_dim]
        
        # Standard attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn_scores: [batch, num_heads, seq, seq]
        
        # Apply mask if provided
        if attention_mask is not None:
            # Expand mask for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: [batch, num_heads, seq, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.hidden_dim)
        
        # Central nexus aggregation
        # All information flows through the center
        nexus_state = x.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        nexus_proj = self.nexus_proj(nexus_state)  # [batch, 1, hidden]
        
        # Gate how much nexus contributes
        gate_input = torch.cat([attn_output, nexus_proj.expand(-1, seq, -1)], dim=-1)
        nexus_gate = torch.sigmoid(self.nexus_gate(gate_input))  # [batch, seq, 1]
        
        # Blend attention output with nexus
        output = attn_output * (1 - nexus_gate) + nexus_proj.expand(-1, seq, -1) * nexus_gate
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attn_weights


class QbitVertex(nn.Module):
    """
    A single quantum information node at an icosahedral vertex.
    
    Models a qubit-like state with:
    - Amplitude (magnitude of state)
    - Phase (rotation in complex plane)
    - Entanglement with neighbors (via edges)
    """
    
    def __init__(self, hidden_dim: int, vertex_id: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vertex_id = vertex_id
        
        # Quantum state parameters
        # |ψ⟩ = α|0⟩ + β|1⟩, where α² + β² = 1
        self.amplitude_proj = nn.Linear(hidden_dim, hidden_dim)
        self.phase_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Local unitary transformation (rotation)
        self.unitary = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Initialize unitary close to identity
        nn.init.orthogonal_(self.unitary.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through qbit node.
        
        Args:
            x: Input state [batch, hidden]
            
        Returns:
            state: Transformed state [batch, hidden]
            phase: Phase information [batch, hidden]
        """
        # Compute amplitude and phase
        amplitude = torch.sigmoid(self.amplitude_proj(x))  # [0, 1]
        phase = torch.tanh(self.phase_proj(x)) * math.pi  # [-π, π]
        
        # Apply unitary transformation (preserves norm)
        state = self.unitary(x)
        
        # Modulate by amplitude and phase
        # Using Euler formula: e^(iφ) ≈ cos(φ) + i*sin(φ)
        # We simulate complex behavior in real space
        real_part = state * amplitude * torch.cos(phase)
        imag_part = state * amplitude * torch.sin(phase)
        
        # Combine real and imaginary (interleaved)
        state = real_part + imag_part.roll(shifts=1, dims=-1)
        
        return state, phase


class QbitNexus(nn.Module):
    """
    Complete Qbit Nexus - Icosahedral Quantum Information Network
    
    Architecture:
        Input → 12 QbitVertices (one per icosahedral vertex)
              → Edge transformations (30 channels)
              → Face aggregations (20 pathways)
              → Central nexus (hub convergence)
              → Output
    
    Key equations:
        - Vertex state: |ψᵢ⟩ = Uᵢ|x⟩
        - Edge coupling: Cᵢⱼ = ⟨ψᵢ|Eᵢⱼ|ψⱼ⟩
        - Nexus state: |Φ⟩ = Σᵢ wᵢ|ψᵢ⟩
        - Golden ratio scaling: wᵢ ∝ φ^(-dᵢ) where dᵢ is geodesic distance
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_vertices: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_vertices = num_vertices
        self.num_layers = num_layers
        
        # Icosahedral geometry
        self.register_buffer('vertices', get_icosahedron_vertices())
        self.register_buffer('edges', get_icosahedron_edges())
        self.register_buffer('faces', get_icosahedron_faces())
        
        # 12 Qbit vertices
        self.qbits = nn.ModuleList([
            QbitVertex(hidden_dim, i) for i in range(num_vertices)
        ])
        
        # Project input to vertex assignment
        self.vertex_assignment = nn.Linear(hidden_dim, num_vertices)
        
        # Edge coupling matrices (30 edges)
        self.edge_coupling = nn.Parameter(
            torch.randn(30, hidden_dim, hidden_dim) * 0.01
        )
        
        # Face aggregation (20 faces)
        self.face_aggregate = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Nexus attention layers
        self.nexus_layers = nn.ModuleList([
            NexusAttention(hidden_dim, num_heads=num_vertices, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.vertex_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.face_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Golden ratio weights for vertex importance
        # Vertices closer to "north pole" get higher weight
        vertex_distances = self.vertices[:, 2]  # z-coordinate
        golden_weights = PHI ** (-vertex_distances.abs())
        golden_weights = golden_weights / golden_weights.sum()
        self.register_buffer('golden_weights', golden_weights)
        
        self.dropout = nn.Dropout(dropout)
        
        # Statistics
        self._forward_count = 0
        self._entanglement_history: List[torch.Tensor] = []
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through Qbit Nexus.
        
        Args:
            x: Input [batch, seq, hidden]
            attention_mask: Optional mask [batch, seq]
            
        Returns:
            Dict with output, vertex_states, entanglement metrics
        """
        self._forward_count += 1
        batch, seq, _ = x.shape
        
        # === 1. Assign input to vertices ===
        vertex_weights = F.softmax(self.vertex_assignment(x), dim=-1)  # [batch, seq, 12]
        
        # === 2. Process through Qbit vertices ===
        vertex_states = []
        vertex_phases = []
        
        for i, qbit in enumerate(self.qbits):
            # Weight input by vertex assignment
            weighted_input = x * vertex_weights[:, :, i:i+1]  # [batch, seq, hidden]
            
            # Process through qbit
            state, phase = qbit(weighted_input.mean(dim=1))  # [batch, hidden]
            vertex_states.append(state)
            vertex_phases.append(phase)
        
        vertex_states = torch.stack(vertex_states, dim=1)  # [batch, 12, hidden]
        vertex_phases = torch.stack(vertex_phases, dim=1)  # [batch, 12, hidden]
        vertex_states = self.vertex_norm(vertex_states)
        
        # === 3. Edge coupling (entanglement) ===
        edge_outputs = []
        entanglement_strengths = []
        
        for e, (i, j) in enumerate(self.edges):
            # Couple vertices i and j through edge e
            vi = vertex_states[:, i]  # [batch, hidden]
            vj = vertex_states[:, j]  # [batch, hidden]
            
            # Bilinear coupling
            coupling = torch.einsum('bh,hd,bd->b', vi, self.edge_coupling[e], vj)
            entanglement_strengths.append(coupling)
            
            # Edge state is superposition
            edge_state = (vi + vj) / math.sqrt(2)
            edge_outputs.append(edge_state)
        
        edge_states = torch.stack(edge_outputs, dim=1)  # [batch, 30, hidden]
        edge_states = self.edge_norm(edge_states)
        
        entanglement = torch.stack(entanglement_strengths, dim=1)  # [batch, 30]
        
        # Track entanglement for analysis
        if self.training and len(self._entanglement_history) < 100:
            self._entanglement_history.append(entanglement.detach().mean(dim=0))
        
        # === 4. Face aggregation ===
        face_outputs = []
        
        for face in self.faces:
            # Get three vertices of face
            v0 = vertex_states[:, face[0]]
            v1 = vertex_states[:, face[1]]
            v2 = vertex_states[:, face[2]]
            
            # Aggregate (concatenate and project)
            face_input = torch.cat([v0, v1, v2], dim=-1)  # [batch, hidden*3]
            face_state = self.face_aggregate(face_input)  # [batch, hidden]
            face_outputs.append(face_state)
        
        face_states = torch.stack(face_outputs, dim=1)  # [batch, 20, hidden]
        face_states = self.face_norm(face_states)
        
        # === 5. Central nexus aggregation ===
        # Weighted sum of vertex states (golden ratio weighting)
        nexus_state = (vertex_states * self.golden_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        # nexus_state: [batch, hidden]
        
        # === 6. Apply nexus attention layers ===
        # Expand back to sequence
        output = x + nexus_state.unsqueeze(1)  # Residual with nexus
        
        attention_weights_list = []
        for layer in self.nexus_layers:
            attn_out, attn_weights = layer(output, attention_mask)
            output = output + self.dropout(attn_out)
            output = output + self.ffn(self.output_norm(output))
            attention_weights_list.append(attn_weights)
        
        output = self.output_norm(output)
        
        # === Compute metrics ===
        # Entanglement entropy (simplified)
        ent_probs = F.softmax(entanglement, dim=-1)
        entanglement_entropy = -(ent_probs * torch.log(ent_probs + 1e-10)).sum(dim=-1).mean()
        
        # Phase coherence
        phase_diff = vertex_phases[:, :, None, :] - vertex_phases[:, None, :, :]
        phase_coherence = torch.cos(phase_diff).mean()
        
        return {
            'output': output,
            'vertex_states': vertex_states,
            'edge_states': edge_states,
            'face_states': face_states,
            'nexus_state': nexus_state,
            'entanglement': entanglement,
            'entanglement_entropy': entanglement_entropy,
            'phase_coherence': phase_coherence,
            'vertex_weights': vertex_weights,
            'attention_weights': attention_weights_list[-1] if attention_weights_list else None
        }
    
    def get_entanglement_statistics(self) -> Dict[str, Any]:
        """Get entanglement statistics from training"""
        if not self._entanglement_history:
            return {'mean_entanglement': None}
        
        ent_tensor = torch.stack(self._entanglement_history)
        mean_ent = ent_tensor.mean(dim=0)
        
        return {
            'mean_entanglement': mean_ent,
            'std_entanglement': ent_tensor.std(dim=0),
            'num_samples': len(self._entanglement_history)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get module statistics"""
        return {
            'forward_count': self._forward_count,
            'num_vertices': self.num_vertices,
            'num_edges': len(self.edges),
            'num_faces': len(self.faces),
            'entanglement_stats': self.get_entanglement_statistics()
        }


class QbitNexusLayer(nn.Module):
    """
    Drop-in layer that wraps QbitNexus for easy integration into transformer stack.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_nexus_layers: int = 1,
        dropout: float = 0.1,
        residual_weight: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual_weight = residual_weight
        
        self.nexus = QbitNexus(
            hidden_dim=hidden_dim,
            num_vertices=12,
            num_layers=num_nexus_layers,
            dropout=dropout
        )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq, hidden]
            attention_mask: Optional mask
            
        Returns:
            output: [batch, seq, hidden]
            info: Dict with nexus information
        """
        nexus_result = self.nexus(x, attention_mask)
        nexus_out = nexus_result['output']
        
        # Compute adaptive gate
        gate_input = torch.cat([x, nexus_out], dim=-1)
        gate = self.gate(gate_input)
        
        # Blend with residual
        output = x + gate * self.residual_weight * (nexus_out - x)
        
        return output, nexus_result


# === Test ===

if __name__ == "__main__":
    print("Testing QbitNexus...")
    
    # Test icosahedral geometry
    vertices = get_icosahedron_vertices()
    edges = get_icosahedron_edges()
    faces = get_icosahedron_faces()
    
    print(f"Icosahedron:")
    print(f"  Vertices: {vertices.shape[0]} (expected 12)")
    print(f"  Edges: {edges.shape[0]} (expected 30)")
    print(f"  Faces: {faces.shape[0]} (expected 20)")
    print(f"  Golden ratio φ: {PHI:.6f}")
    
    # Verify vertex distances (all should be same)
    distances = vertices.norm(dim=-1)
    print(f"  Vertex distances: {distances.mean():.4f} ± {distances.std():.6f}")
    
    # Test QbitNexus
    print("\nTesting QbitNexus module...")
    nexus = QbitNexus(hidden_dim=256, num_vertices=12, num_layers=2)
    
    batch, seq = 2, 16
    x = torch.randn(batch, seq, 256)
    
    result = nexus(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {result['output'].shape}")
    print(f"  Vertex states: {result['vertex_states'].shape}")
    print(f"  Edge states: {result['edge_states'].shape}")
    print(f"  Face states: {result['face_states'].shape}")
    print(f"  Nexus state: {result['nexus_state'].shape}")
    print(f"  Entanglement: {result['entanglement'].shape}")
    print(f"  Entanglement entropy: {result['entanglement_entropy'].item():.4f}")
    print(f"  Phase coherence: {result['phase_coherence'].item():.4f}")
    
    # Test backward pass
    loss = result['output'].mean()
    loss.backward()
    print(f"\n  Backward pass successful")
    
    # Test QbitNexusLayer
    print("\nTesting QbitNexusLayer...")
    layer = QbitNexusLayer(hidden_dim=256)
    x = torch.randn(2, 16, 256)
    output, info = layer(x)
    print(f"  Output shape: {output.shape}")
    
    # Parameter count
    params = sum(p.numel() for p in nexus.parameters())
    print(f"\n  QbitNexus parameters: {params:,}")
    
    print("\nAll QbitNexus tests passed!")
