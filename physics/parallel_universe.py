"""
Parallel Universe / Overlapping Dimensions Theory

Models multiple parallel computational "universes" that overlap and interfere,
inspired by:
- Many-Worlds Interpretation (Everett)
- Multiverse branching at decision points
- Dimensional overlap / interference patterns
- Quantum superposition of parallel states

Key Concepts:
1. PARALLEL UNIVERSES: Multiple independent processing streams
2. DIMENSIONAL OVERLAP: Universes share certain dimensions, diverge in others
3. INTERFERENCE: When universes overlap, they can constructively/destructively interfere
4. BRANCHING: At decision points, universes split and evolve independently
5. COLLAPSE: Observation/measurement causes universe selection

Neural Architecture Mapping:
- Each "universe" is a parallel attention head or processing pathway
- Overlapping dimensions are shared embedding subspaces
- Interference is modeled via complex-valued or phase-aware attention
- Branching occurs at gating mechanisms
- Collapse is the final aggregation/selection

Mathematical Foundation:
- State in universe i: |ψᵢ⟩ ∈ ℋᵢ
- Overlap operator: Oᵢⱼ = ⟨ψᵢ|ψⱼ⟩ (inner product measures overlap)
- Interference: |ψ_total|² = Σᵢ|ψᵢ|² + Σᵢ≠ⱼ ψᵢ*ψⱼ (cross terms)
- Dimension projection: Πₖ projects onto shared dimension k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math


class ParallelUniverse(nn.Module):
    """
    A single parallel universe - an independent computational pathway.
    
    Each universe has:
    - Its own "laws of physics" (learned transformations)
    - Partial dimensional overlap with other universes
    - Phase information for interference
    """
    
    def __init__(
        self,
        hidden_dim: int,
        universe_id: int,
        num_dimensions: int = 8,  # Total dimensions in multiverse
        overlap_dims: int = 4,    # Dimensions shared across universes
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.universe_id = universe_id
        self.num_dimensions = num_dimensions
        self.overlap_dims = overlap_dims
        self.private_dims = num_dimensions - overlap_dims
        
        self.dim_per_component = hidden_dim // num_dimensions
        
        # Universe-specific "laws" (transformations)
        self.laws = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Phase generator (for interference)
        self.phase_net = nn.Linear(hidden_dim, hidden_dim)
        
        # Dimensional projectors
        # Shared dimensions: same across universes
        # Private dimensions: unique to this universe
        self.shared_proj = nn.Linear(hidden_dim, overlap_dims * self.dim_per_component)
        self.private_proj = nn.Linear(hidden_dim, self.private_dims * self.dim_per_component)
        
        # Universe "constants" (learnable parameters unique to this universe)
        self.constants = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evolve state through this universe's laws.
        
        Args:
            x: Input state [batch, seq, hidden]
            
        Returns:
            Dict with evolved state, phase, shared/private components
        """
        # Add universe constants (unique "physics")
        x_const = x + self.constants
        
        # Apply universe laws
        evolved = self.laws(x_const)
        evolved = self.norm(evolved + x)  # Residual
        
        # Compute phase for interference
        phase = torch.tanh(self.phase_net(evolved)) * math.pi  # [-π, π]
        
        # Project to shared and private dimensions
        shared = self.shared_proj(evolved)  # Dimensions that overlap
        private = self.private_proj(evolved)  # Dimensions unique to this universe
        
        return {
            'state': evolved,
            'phase': phase,
            'shared': shared,
            'private': private,
            'amplitude': torch.norm(evolved, dim=-1, keepdim=True)
        }


class DimensionalOverlap(nn.Module):
    """
    Models the overlap between parallel universes.
    
    When universes share dimensions, their states can:
    - Constructively interfere (amplify)
    - Destructively interfere (cancel)
    - Create entanglement between universes
    """
    
    def __init__(self, hidden_dim: int, num_universes: int, overlap_dims: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_universes = num_universes
        self.overlap_dims = overlap_dims
        
        # Overlap strength matrix (learnable)
        # How much do universes i and j overlap?
        self.overlap_matrix = nn.Parameter(
            torch.eye(num_universes) * 0.5 + torch.randn(num_universes, num_universes) * 0.1
        )
        
        # Phase alignment network
        self.phase_align = nn.Linear(hidden_dim * 2, 1)
        
        # Interference combiner
        self.combine = nn.Linear(hidden_dim * num_universes, hidden_dim)
    
    def compute_overlap(
        self,
        shared_states: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute overlap matrix between universe shared dimensions.
        
        Args:
            shared_states: List of [batch, seq, shared_dim] tensors
            
        Returns:
            overlap: [batch, num_universes, num_universes] overlap strengths
        """
        batch = shared_states[0].shape[0]
        seq = shared_states[0].shape[1]
        
        # Stack: [batch, seq, num_universes, shared_dim]
        stacked = torch.stack(shared_states, dim=2)
        
        # Normalize for cosine similarity
        normed = F.normalize(stacked, dim=-1)
        
        # Compute pairwise overlaps: [batch, seq, num_universes, num_universes]
        overlap = torch.einsum('bsid,bsjd->bsij', normed, normed)
        
        # Modulate by learned overlap matrix
        overlap = overlap * torch.sigmoid(self.overlap_matrix)
        
        # Average over sequence
        overlap = overlap.mean(dim=1)  # [batch, num_universes, num_universes]
        
        return overlap
    
    def compute_interference(
        self,
        states: List[torch.Tensor],
        phases: List[torch.Tensor],
        overlap: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interference pattern between universes.
        
        Interference = Σᵢ|ψᵢ|² + Σᵢ≠ⱼ |ψᵢ||ψⱼ|cos(φᵢ - φⱼ)
        
        Args:
            states: List of [batch, seq, hidden] universe states
            phases: List of [batch, seq, hidden] phase tensors
            overlap: [batch, num_universes, num_universes] overlap strengths
            
        Returns:
            interference: [batch, seq, hidden] interference pattern
        """
        batch, seq, hidden = states[0].shape
        n = len(states)
        
        # Self-interference (intensity) terms
        intensities = [s.pow(2) for s in states]
        total_intensity = sum(intensities)
        
        # Cross-interference terms
        cross_terms = torch.zeros_like(total_intensity)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Phase difference
                phase_diff = phases[i] - phases[j]
                
                # Amplitude product
                amp_product = states[i].abs() * states[j].abs()
                
                # Interference: cos(Δφ) gives constructive (+1) or destructive (-1)
                interference = amp_product * torch.cos(phase_diff)
                
                # Weight by overlap strength
                overlap_ij = overlap[:, i, j].unsqueeze(1).unsqueeze(2)
                cross_terms = cross_terms + 2 * overlap_ij * interference
        
        # Total interference pattern
        interference_pattern = total_intensity + cross_terms
        
        return interference_pattern
    
    def forward(
        self,
        universe_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """
        Combine universe outputs through dimensional overlap.
        
        Args:
            universe_outputs: List of universe output dicts
            
        Returns:
            Dict with combined state and interference info
        """
        states = [u['state'] for u in universe_outputs]
        phases = [u['phase'] for u in universe_outputs]
        shared = [u['shared'] for u in universe_outputs]
        
        # Compute overlap
        overlap = self.compute_overlap(shared)
        
        # Compute interference
        interference = self.compute_interference(states, phases, overlap)
        
        # Combine all states (with interference modulation)
        stacked = torch.stack(states, dim=-1)  # [batch, seq, hidden, n]
        combined = stacked.sum(dim=-1) / math.sqrt(len(states))
        
        # Modulate by interference pattern
        combined = combined * torch.sigmoid(interference)
        
        # Final projection
        all_states = torch.cat(states, dim=-1)  # [batch, seq, hidden * n]
        combined = self.combine(all_states) + combined
        
        return {
            'combined': combined,
            'overlap_matrix': overlap,
            'interference_pattern': interference,
            'individual_states': states
        }


class UniverseBranching(nn.Module):
    """
    Models universe branching at decision points.
    
    At certain "measurement" points, universes can:
    - Split into multiple branches
    - Merge back together
    - Exchange information through entanglement
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_universes: int,
        branching_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_universes = num_universes
        self.branching_factor = branching_factor
        
        # Branching decision network
        self.branch_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, branching_factor),
            nn.Softmax(dim=-1)
        )
        
        # Branch-specific transformations
        self.branch_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(branching_factor)
        ])
        
        # Merge network
        self.merge = nn.Linear(hidden_dim * branching_factor, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def branch(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Branch universe into multiple paths.
        
        Args:
            x: Input state [batch, seq, hidden]
            
        Returns:
            branches: List of [batch, seq, hidden] branch states
            branch_weights: [batch, seq, branching_factor] selection weights
        """
        # Compute branching weights
        branch_weights = self.branch_gate(x)  # [batch, seq, branching_factor]
        
        # Apply branch-specific transforms
        branches = []
        for i, transform in enumerate(self.branch_transforms):
            branch = transform(x) * branch_weights[:, :, i:i+1]
            branches.append(branch)
        
        return branches, branch_weights
    
    def merge_branches(self, branches: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge branches back into single universe.
        
        Args:
            branches: List of [batch, seq, hidden] branch states
            
        Returns:
            merged: [batch, seq, hidden] merged state
        """
        concatenated = torch.cat(branches, dim=-1)
        merged = self.merge(concatenated)
        return self.dropout(merged)
    
    def forward(
        self,
        x: torch.Tensor,
        do_branch: bool = True,
        do_merge: bool = True
    ) -> Dict[str, Any]:
        """
        Process through branching mechanism.
        
        Args:
            x: Input [batch, seq, hidden]
            do_branch: Whether to branch
            do_merge: Whether to merge branches back
            
        Returns:
            Dict with output and branching info
        """
        if do_branch:
            branches, weights = self.branch(x)
            
            if do_merge:
                output = self.merge_branches(branches)
            else:
                output = branches  # Return list of branches
        else:
            output = x
            branches = [x]
            weights = torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
        
        return {
            'output': output,
            'branches': branches if not do_merge else None,
            'branch_weights': weights
        }


class MultiverseCollapse(nn.Module):
    """
    Models wavefunction collapse / universe selection.
    
    At the end, we must select/collapse to a single output.
    This can be:
    - Soft collapse (weighted average)
    - Hard collapse (argmax selection)
    - Measurement-based (learned observation operator)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_universes: int,
        collapse_mode: str = 'soft'  # 'soft', 'hard', 'measured'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_universes = num_universes
        self.collapse_mode = collapse_mode
        
        # Observation/measurement operator
        self.measure = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Universe selection network
        self.selector = nn.Linear(hidden_dim * num_universes, num_universes)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        universe_states: List[torch.Tensor],
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Collapse multiverse to single output.
        
        Args:
            universe_states: List of [batch, seq, hidden] states
            temperature: Softmax temperature (lower = harder collapse)
            
        Returns:
            Dict with collapsed output and selection info
        """
        batch, seq, hidden = universe_states[0].shape
        n = len(universe_states)
        
        # Compute "measurement" values for each universe
        measurements = []
        for state in universe_states:
            m = self.measure(state)  # [batch, seq, 1]
            measurements.append(m)
        
        measurements = torch.cat(measurements, dim=-1)  # [batch, seq, n]
        
        # Compute selection probabilities
        if self.collapse_mode == 'soft':
            probs = F.softmax(measurements / temperature, dim=-1)
        elif self.collapse_mode == 'hard':
            # Gumbel-softmax for differentiable hard selection
            probs = F.gumbel_softmax(measurements, tau=temperature, hard=True)
        else:  # 'measured'
            # Use full context for selection
            stacked = torch.cat(universe_states, dim=-1)  # [batch, seq, hidden * n]
            logits = self.selector(stacked)  # [batch, seq, n]
            probs = F.softmax(logits / temperature, dim=-1)
        
        # Collapse: weighted sum of universe states
        stacked = torch.stack(universe_states, dim=-1)  # [batch, seq, hidden, n]
        collapsed = (stacked * probs.unsqueeze(2)).sum(dim=-1)  # [batch, seq, hidden]
        
        # Output projection
        output = self.output_proj(collapsed)
        
        # Compute entropy of selection (uncertainty)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        return {
            'output': output,
            'selection_probs': probs,
            'entropy': entropy,
            'measurements': measurements
        }


class ParallelUniverseLayer(nn.Module):
    """
    Complete Parallel Universe / Overlapping Dimensions layer.
    
    Architecture:
        Input → Branch into N parallel universes
              → Each universe evolves independently
              → Dimensional overlap creates interference
              → Collapse back to single output
    
    This models:
    - Parallel hypothesis exploration
    - Multi-scale feature processing  
    - Ensemble-like computation
    - Quantum-inspired superposition
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_universes: int = 4,
        num_dimensions: int = 8,
        overlap_dims: int = 4,
        collapse_mode: str = 'soft',
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_universes = num_universes
        
        # Create parallel universes
        self.universes = nn.ModuleList([
            ParallelUniverse(
                hidden_dim=hidden_dim,
                universe_id=i,
                num_dimensions=num_dimensions,
                overlap_dims=overlap_dims,
                dropout=dropout
            ) for i in range(num_universes)
        ])
        
        # Dimensional overlap handler
        self.overlap = DimensionalOverlap(
            hidden_dim=hidden_dim,
            num_universes=num_universes,
            overlap_dims=overlap_dims
        )
        
        # Universe branching
        self.branching = UniverseBranching(
            hidden_dim=hidden_dim,
            num_universes=num_universes,
            dropout=dropout
        )
        
        # Multiverse collapse
        self.collapse = MultiverseCollapse(
            hidden_dim=hidden_dim,
            num_universes=num_universes,
            collapse_mode=collapse_mode
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Statistics
        self._forward_count = 0
        self._interference_history: List[torch.Tensor] = []
    
    def forward(
        self,
        x: torch.Tensor,
        collapse_temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Forward pass through parallel universes.
        
        Args:
            x: Input [batch, seq, hidden]
            collapse_temperature: Temperature for collapse (lower = more decisive)
            
        Returns:
            Dict with output and multiverse info
        """
        self._forward_count += 1
        
        # === 1. Evolve through each universe ===
        universe_outputs = []
        for universe in self.universes:
            output = universe(x)
            universe_outputs.append(output)
        
        # === 2. Compute dimensional overlap and interference ===
        overlap_result = self.overlap(universe_outputs)
        
        # Track interference for analysis
        if self.training:
            interference_mean = overlap_result['interference_pattern'].mean(dim=(0, 1))
            if len(self._interference_history) < 100:
                self._interference_history.append(interference_mean.detach())
        
        # === 3. Apply branching within combined state ===
        branch_result = self.branching(overlap_result['combined'])
        
        # === 4. Collapse to single output ===
        universe_states = [u['state'] for u in universe_outputs]
        collapse_result = self.collapse(universe_states, temperature=collapse_temperature)
        
        # Combine collapsed output with branched/interfered state
        output = self.norm(
            collapse_result['output'] + 
            0.5 * branch_result['output'] + 
            0.3 * overlap_result['combined']
        )
        
        # Add residual
        output = output + x
        
        return {
            'output': output,
            'universe_outputs': universe_outputs,
            'overlap_matrix': overlap_result['overlap_matrix'],
            'interference_pattern': overlap_result['interference_pattern'],
            'branch_weights': branch_result['branch_weights'],
            'selection_probs': collapse_result['selection_probs'],
            'collapse_entropy': collapse_result['entropy']
        }
    
    def get_interference_statistics(self) -> Dict[str, Any]:
        """Get interference pattern statistics"""
        if not self._interference_history:
            return {'mean_interference': None}
        
        interference = torch.stack(self._interference_history)
        return {
            'mean_interference': interference.mean(dim=0),
            'std_interference': interference.std(dim=0),
            'num_samples': len(self._interference_history)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics"""
        return {
            'forward_count': self._forward_count,
            'num_universes': self.num_universes,
            'interference_stats': self.get_interference_statistics()
        }


class MultiverseAttention(nn.Module):
    """
    Attention mechanism where each head operates in a different "universe".
    
    Heads can have:
    - Different "laws" (different projections)
    - Overlapping dimensions (shared key/value subspaces)
    - Interference between heads
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        overlap_fraction: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.overlap_fraction = overlap_fraction
        
        self.overlap_dim = int(self.head_dim * overlap_fraction)
        self.private_dim = self.head_dim - self.overlap_dim
        
        # Shared Q, K, V projections (overlap dimensions)
        self.shared_q = nn.Linear(hidden_dim, self.overlap_dim * num_heads)
        self.shared_k = nn.Linear(hidden_dim, self.overlap_dim * num_heads)
        self.shared_v = nn.Linear(hidden_dim, self.overlap_dim * num_heads)
        
        # Private Q, K, V projections (per-head dimensions)
        self.private_qkv = nn.ModuleList([
            nn.Linear(hidden_dim, self.private_dim * 3) for _ in range(num_heads)
        ])
        
        # Phase for interference
        self.phase_proj = nn.Linear(hidden_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with multiverse attention.
        
        Args:
            x: Input [batch, seq, hidden]
            attention_mask: Optional mask [batch, seq]
            
        Returns:
            output: [batch, seq, hidden]
            info: Dict with attention info
        """
        batch, seq, _ = x.shape
        
        # Shared projections
        shared_q = self.shared_q(x).view(batch, seq, self.num_heads, self.overlap_dim)
        shared_k = self.shared_k(x).view(batch, seq, self.num_heads, self.overlap_dim)
        shared_v = self.shared_v(x).view(batch, seq, self.num_heads, self.overlap_dim)
        
        # Private projections (different for each head/universe)
        private_qs, private_ks, private_vs = [], [], []
        for i, proj in enumerate(self.private_qkv):
            qkv = proj(x)  # [batch, seq, private_dim * 3]
            q, k, v = qkv.chunk(3, dim=-1)
            private_qs.append(q)
            private_ks.append(k)
            private_vs.append(v)
        
        private_q = torch.stack(private_qs, dim=2)  # [batch, seq, num_heads, private_dim]
        private_k = torch.stack(private_ks, dim=2)
        private_v = torch.stack(private_vs, dim=2)
        
        # Combine shared and private
        q = torch.cat([shared_q, private_q], dim=-1)  # [batch, seq, num_heads, head_dim]
        k = torch.cat([shared_k, private_k], dim=-1)
        v = torch.cat([shared_v, private_v], dim=-1)
        
        # Transpose for attention: [batch, num_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Compute phase for interference between heads
        phases = self.phase_proj(x)  # [batch, seq, num_heads]
        phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq, num_heads, num_heads]
        
        # Interference term (add to attention between heads that are "in phase")
        # This is applied across the head dimension implicitly through the output combination
        
        # Apply mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq, head_dim]
        
        # Apply phase-based interference to output
        # Heads with similar phases constructively interfere
        phase_interference = torch.cos(phases).unsqueeze(-1)  # [batch, seq, num_heads, 1]
        attn_output = attn_output.transpose(1, 2)  # [batch, seq, num_heads, head_dim]
        attn_output = attn_output * (1 + 0.1 * phase_interference)
        
        # Reshape and project
        attn_output = attn_output.contiguous().view(batch, seq, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        return output, {
            'attention_weights': attn_weights,
            'phases': phases,
            'overlap_fraction': self.overlap_fraction
        }


# === Test ===

if __name__ == "__main__":
    print("Testing Parallel Universe / Overlapping Dimensions...")
    
    batch, seq, hidden = 2, 16, 256
    x = torch.randn(batch, seq, hidden)
    
    # Test ParallelUniverse
    print("\n1. Testing ParallelUniverse...")
    universe = ParallelUniverse(hidden_dim=256, universe_id=0)
    result = universe(x)
    print(f"   State shape: {result['state'].shape}")
    print(f"   Phase shape: {result['phase'].shape}")
    print(f"   Shared dims: {result['shared'].shape}")
    print(f"   Private dims: {result['private'].shape}")
    
    # Test DimensionalOverlap
    print("\n2. Testing DimensionalOverlap...")
    overlap = DimensionalOverlap(hidden_dim=256, num_universes=4, overlap_dims=4)
    universe_outputs = [ParallelUniverse(256, i)(x) for i in range(4)]
    overlap_result = overlap(universe_outputs)
    print(f"   Combined shape: {overlap_result['combined'].shape}")
    print(f"   Overlap matrix: {overlap_result['overlap_matrix'].shape}")
    print(f"   Interference: {overlap_result['interference_pattern'].shape}")
    
    # Test UniverseBranching
    print("\n3. Testing UniverseBranching...")
    branching = UniverseBranching(hidden_dim=256, num_universes=4)
    branch_result = branching(x)
    print(f"   Output shape: {branch_result['output'].shape}")
    print(f"   Branch weights: {branch_result['branch_weights'].shape}")
    
    # Test MultiverseCollapse
    print("\n4. Testing MultiverseCollapse...")
    collapse = MultiverseCollapse(hidden_dim=256, num_universes=4)
    states = [torch.randn(batch, seq, hidden) for _ in range(4)]
    collapse_result = collapse(states)
    print(f"   Output shape: {collapse_result['output'].shape}")
    print(f"   Selection probs: {collapse_result['selection_probs'].shape}")
    print(f"   Entropy: {collapse_result['entropy'].item():.4f}")
    
    # Test full ParallelUniverseLayer
    print("\n5. Testing ParallelUniverseLayer...")
    layer = ParallelUniverseLayer(
        hidden_dim=256,
        num_universes=4,
        num_dimensions=8,
        overlap_dims=4
    )
    result = layer(x)
    print(f"   Output shape: {result['output'].shape}")
    print(f"   Overlap matrix: {result['overlap_matrix'].shape}")
    print(f"   Collapse entropy: {result['collapse_entropy'].item():.4f}")
    
    # Test MultiverseAttention
    print("\n6. Testing MultiverseAttention...")
    attn = MultiverseAttention(hidden_dim=256, num_heads=8, overlap_fraction=0.5)
    output, info = attn(x)
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights: {info['attention_weights'].shape}")
    print(f"   Phases: {info['phases'].shape}")
    
    # Test backward pass
    print("\n7. Testing backward pass...")
    loss = result['output'].mean()
    loss.backward()
    print("   Backward pass successful!")
    
    # Parameter count
    params = sum(p.numel() for p in layer.parameters())
    print(f"\n   ParallelUniverseLayer parameters: {params:,}")
    
    print("\nAll Parallel Universe tests passed!")
