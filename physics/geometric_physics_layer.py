"""
Unified Geometric Physics Layer
Combines all advanced physics and geometry theories into a single neural module

Integrates:
1. Fuller Synergetics - Vector Equilibrium, Tensegrity, Geodesic Frequency
2. Lloyd Computational Universe - Computational limits, Landauer costs, quantum evolution
3. Morphogenesis Dynamics - Turing patterns, Ricci flow, catastrophe theory
4. TPMS Attention - Gyroid/Schwarz minimal surface guided attention
5. Quantum Coupling - Harmonic oscillator dynamics for limb coordination

Architecture:
    Input (hidden states)
         ↓
    ┌─────────────────────────────────────────────────────────┐
    │              GeometricPhysicsLayer                       │
    │                                                          │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
    │  │  Fuller  │  │  Lloyd   │  │  Morpho  │  │   TPMS   │ │
    │  │Synergetic│  │Computat. │  │ genesis  │  │Attention │ │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
    │       │             │             │             │        │
    │       └──────┬──────┴──────┬──────┴──────┬──────┘        │
    │              │             │             │               │
    │         [Learnable Gating / Combination]                 │
    │                        │                                 │
    │              [Physics-Informed Loss]                     │
    └─────────────────────────────────────────────────────────┘
         ↓
    Output (enhanced hidden states + physics losses)

Key insight: Optimal neural architectures emerge from fundamental physics/geometry -
minimal surfaces (TPMS), tensegrity (Fuller), computational limits (Lloyd),
and morphogenesis (Turing) all describe efficient information processing in nature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import math

# Import the advanced modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.fuller_synergetics import FullerSynergetics
from physics.lloyd_computational_universe import LloydComputationalUniverse
from geometry.morphogenesis import MorphogenesisDynamics
from sync.tpms_attention import TPMSAttention
from geometry.qbit_nexus import QbitNexus
from physics.parallel_universe import ParallelUniverseLayer


@dataclass
class GeometricPhysicsConfig:
    """Configuration for the Geometric Physics Layer"""
    # Module enables
    enable_fuller: bool = True
    enable_lloyd: bool = True
    enable_morphogenesis: bool = True
    enable_tpms: bool = True
    enable_qbit_nexus: bool = True
    enable_parallel_universe: bool = True
    
    # Combination mode: 'learnable', 'sequential', 'compound', 'parallel', 'residual'
    # - 'learnable': All modules run in parallel, combined via learned gating
    # - 'compound': Modules feed into each other in a physics-inspired chain
    # - 'sequential': Simple sequential pass through enabled modules
    # - 'parallel': Average of all module outputs
    # - 'residual': Sum of all module outputs with residual connection
    combination_mode: str = 'learnable'
    
    # Fuller Synergetics config
    fuller_ve_vertices: int = 12
    fuller_tensegrity_struts: int = 6
    fuller_geodesic_frequency: int = 2
    
    # Lloyd Computational Universe config
    lloyd_energy_budget: float = 1.0
    lloyd_temperature: float = 1.0
    lloyd_reversible_fraction: float = 0.5
    
    # Morphogenesis config
    morpho_diffusion_steps: int = 3
    morpho_activator_diffusion: float = 0.1
    morpho_inhibitor_diffusion: float = 0.4
    
    # TPMS config
    tpms_surface_type: str = 'gyroid'  # gyroid, schwarz_p, schwarz_d, neovius
    tpms_num_heads: int = 8
    tpms_threshold: float = 0.1
    
    # QbitNexus config (icosahedral quantum network)
    qbit_num_vertices: int = 12  # Icosahedron has 12 vertices
    qbit_num_layers: int = 2
    qbit_dropout: float = 0.1
    
    # ParallelUniverse config (multiverse computation)
    parallel_num_universes: int = 4
    parallel_num_dimensions: int = 8
    parallel_overlap_dims: int = 4
    parallel_collapse_mode: str = 'soft'  # 'soft', 'hard', 'superposition'
    
    # Loss weights
    loss_weight_tensegrity: float = 0.01  # Tension-compression balance
    loss_weight_lloyd: float = 0.01  # Computational efficiency
    loss_weight_turing: float = 0.01  # Pattern emergence
    loss_weight_equilibrium: float = 0.01  # Vector equilibrium stability
    loss_weight_entanglement: float = 0.01  # Quantum entanglement coherence
    loss_weight_universe_overlap: float = 0.01  # Universe interference
    
    # Dropout
    dropout: float = 0.1


class LearnableCombiner(nn.Module):
    """
    Learnable combination of multiple module outputs with attention-based gating.
    """
    
    def __init__(self, hidden_dim: int, num_modules: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modules = num_modules
        
        # Learnable gate logits per module
        self.gate_logits = nn.Parameter(torch.zeros(num_modules))
        
        # Context-dependent gating
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_modules)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        module_outputs: List[torch.Tensor],
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine module outputs with learned gating.
        
        Args:
            module_outputs: List of [batch, seq, hidden] tensors
            context: Context for gating [batch, seq, hidden]
            
        Returns:
            combined: [batch, seq, hidden]
            gate_weights: [batch, seq, num_modules]
        """
        # Stack outputs: [batch, seq, hidden, num_modules]
        stacked = torch.stack(module_outputs, dim=-1)
        
        # Compute context-dependent gates
        # Use mean-pooled context for efficiency
        ctx_pooled = context.mean(dim=1)  # [batch, hidden]
        dynamic_gates = self.gate_proj(ctx_pooled)  # [batch, num_modules]
        
        # Combine with static gates
        combined_gates = self.gate_logits + dynamic_gates  # [batch, num_modules]
        gate_weights = F.softmax(combined_gates, dim=-1)  # [batch, num_modules]
        
        # Expand for broadcasting
        gate_weights_expanded = gate_weights.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_modules]
        
        # Weighted combination
        combined = (stacked * gate_weights_expanded).sum(dim=-1)  # [batch, seq, hidden]
        
        # Output projection with residual
        combined = self.norm(self.dropout(self.output_proj(combined)) + context)
        
        # Return expanded gate weights for logging
        batch, seq, _ = context.shape
        gate_weights_full = gate_weights.unsqueeze(1).expand(batch, seq, -1)
        
        return combined, gate_weights_full


class GeometricPhysicsLayer(nn.Module):
    """
    Unified layer combining all geometric and physics-based neural computations.
    
    This layer transforms hidden states through multiple physics-inspired pathways:
    1. Fuller Synergetics: Structural stability via tensegrity and geodesic patterns
    2. Lloyd Computational: Information-theoretic efficiency constraints
    3. Morphogenesis: Pattern emergence via reaction-diffusion dynamics
    4. TPMS Attention: Minimal surface topology for attention routing
    
    All pathways are combined via learnable gating, and physics-informed losses
    encourage the network to respect fundamental constraints.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[GeometricPhysicsConfig] = None,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.config = config or GeometricPhysicsConfig()
        self.num_heads = num_heads
        
        # Track which modules are enabled
        self.enabled_modules = []
        
        # === Fuller Synergetics ===
        if self.config.enable_fuller:
            self.fuller = FullerSynergetics(
                hidden_dim=hidden_dim,
                enable_ve=True,
                enable_tensegrity=True,
                enable_geodesic=True,
                enable_packing=True,
                enable_quadray=True
            )
            self.enabled_modules.append('fuller')
        
        # === Lloyd Computational Universe ===
        if self.config.enable_lloyd:
            self.lloyd = LloydComputationalUniverse(
                hidden_dim=hidden_dim,
                num_qubits=8,
                temperature=self.config.lloyd_temperature,
                energy_budget=self.config.lloyd_energy_budget,
                enable_quantum_evolution=True,
                enable_complexity=True
            )
            self.enabled_modules.append('lloyd')
        
        # === Morphogenesis Dynamics ===
        if self.config.enable_morphogenesis:
            self.morphogenesis = MorphogenesisDynamics(
                hidden_dim=hidden_dim,
                enable_turing=True,
                enable_geometric_flow=True,
                enable_catastrophe=True,
                enable_fermentation=True,
                enable_symplectic=True
            )
            self.enabled_modules.append('morphogenesis')
        
        # === TPMS Attention ===
        if self.config.enable_tpms:
            self.tpms = TPMSAttention(
                hidden_dim=hidden_dim,
                num_heads=self.config.tpms_num_heads,
                tpms_type=self.config.tpms_surface_type,
                tpms_weight=0.3,
                dropout=self.config.dropout
            )
            self.enabled_modules.append('tpms')
        
        # === QbitNexus (Icosahedral Quantum Network) ===
        # Inspired by D-Wave's quantum annealing and Geordie Rose's vision:
        # Information flows through icosahedral geometry (12 vertices, golden ratio)
        # like qubits in a quantum computer, enabling superposition of features
        if self.config.enable_qbit_nexus:
            self.qbit_nexus = QbitNexus(
                hidden_dim=hidden_dim,
                num_vertices=self.config.qbit_num_vertices,
                num_layers=self.config.qbit_num_layers,
                dropout=self.config.qbit_dropout
            )
            self.enabled_modules.append('qbit_nexus')
        
        # === ParallelUniverseLayer (Multiverse Computation) ===
        # Multiple computational "universes" process in parallel,
        # then collapse via interference - inspired by many-worlds interpretation
        # and the idea that intelligence explores possibility space
        if self.config.enable_parallel_universe:
            self.parallel_universe = ParallelUniverseLayer(
                hidden_dim=hidden_dim,
                num_universes=self.config.parallel_num_universes,
                num_dimensions=self.config.parallel_num_dimensions,
                overlap_dims=self.config.parallel_overlap_dims,
                collapse_mode=self.config.parallel_collapse_mode
            )
            self.enabled_modules.append('parallel_universe')
        
        # === Learnable Combination ===
        num_enabled = len(self.enabled_modules)
        if num_enabled > 1:
            self.combiner = LearnableCombiner(
                hidden_dim=hidden_dim,
                num_modules=num_enabled,
                dropout=self.config.dropout
            )
        else:
            self.combiner = None
        
        # === Physics Loss Accumulators ===
        self._physics_losses: Dict[str, torch.Tensor] = {}
        
        # === Statistics ===
        self._forward_count = 0
        self._gate_history: List[torch.Tensor] = []
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through geometric physics layer.
        
        Args:
            x: Input hidden states [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch, seq_len]
            return_components: Whether to return individual module outputs
            
        Returns:
            Dict containing:
                - output: Enhanced hidden states [batch, seq, hidden]
                - physics_loss: Combined physics-informed loss
                - gate_weights: Module gating weights (if multiple modules)
                - components: Individual module outputs (if return_components)
        """
        self._forward_count += 1
        batch, seq_len, _ = x.shape
        
        component_outputs = {}
        self._physics_losses = {}
        
        # === COMPOUND MODE: Sequential chained processing ===
        if self.config.combination_mode == 'compound':
            combined = self._forward_compound(x, attention_mask, component_outputs)
            gate_weights = None
        else:
            # === PARALLEL MODE: All modules process same input ===
            outputs = []
            
            if self.config.enable_fuller:
                fuller_result = self.fuller(x)
                fuller_out = fuller_result['combined']
                outputs.append(fuller_out)
                component_outputs['fuller'] = fuller_result
                
                if 'tensegrity' in fuller_result:
                    tensegrity_out = fuller_result['tensegrity']
                    tensegrity_loss = tensegrity_out.var(dim=-1).mean()
                    self._physics_losses['tensegrity'] = tensegrity_loss * self.config.loss_weight_tensegrity
            
            if self.config.enable_lloyd:
                lloyd_out, lloyd_info = self.lloyd(x)
                outputs.append(lloyd_out)
                component_outputs['lloyd'] = {'output': lloyd_out, 'info': lloyd_info}
                
                if 'landauer_cost' in lloyd_info:
                    self._physics_losses['lloyd'] = lloyd_info['landauer_cost'] * self.config.loss_weight_lloyd
            
            if self.config.enable_morphogenesis:
                morpho_result = self.morphogenesis(x)
                morpho_out = morpho_result['combined']
                outputs.append(morpho_out)
                component_outputs['morphogenesis'] = morpho_result
                
                if 'turing' in morpho_result:
                    pattern_var = morpho_result['turing'].var(dim=-1).mean()
                    self._physics_losses['turing'] = -torch.log(pattern_var + 1e-6) * self.config.loss_weight_turing
            
            if self.config.enable_tpms:
                tpms_out, tpms_weights = self.tpms(x, mask=attention_mask)
                outputs.append(tpms_out)
                component_outputs['tpms'] = {'output': tpms_out, 'weights': tpms_weights}
            
            if self.config.enable_qbit_nexus:
                qbit_result = self.qbit_nexus(x)
                qbit_out = qbit_result['output']
                outputs.append(qbit_out)
                component_outputs['qbit_nexus'] = qbit_result
                
                if 'entanglement' in qbit_result:
                    entanglement = qbit_result['entanglement']
                    entanglement_magnitude = (entanglement ** 2).mean()
                    self._physics_losses['entanglement'] = -torch.log(entanglement_magnitude + 1e-6) * self.config.loss_weight_entanglement
            
            if self.config.enable_parallel_universe:
                parallel_result = self.parallel_universe(x)
                parallel_out = parallel_result['output']
                outputs.append(parallel_out)
                component_outputs['parallel_universe'] = parallel_result
                
                if 'interference_pattern' in parallel_result:
                    interference_var = parallel_result['interference_pattern'].var()
                    self._physics_losses['universe_overlap'] = -torch.log(interference_var + 1e-6) * self.config.loss_weight_universe_overlap
            
            # === Combine outputs based on mode ===
            if len(outputs) == 0:
                combined = x
                gate_weights = None
            elif len(outputs) == 1:
                combined = outputs[0]
                gate_weights = torch.ones(batch, seq_len, 1, device=x.device)
            elif self.config.combination_mode == 'sequential':
                combined = outputs[-1]
                gate_weights = None
            elif self.config.combination_mode == 'parallel':
                combined = torch.stack(outputs, dim=0).mean(dim=0)
                gate_weights = None
            elif self.config.combination_mode == 'residual':
                combined = x + sum(outputs)
                gate_weights = None
            else:  # 'learnable'
                combined, gate_weights = self.combiner(outputs, x)
                if self.training and len(self._gate_history) < 100:
                    self._gate_history.append(gate_weights.detach().mean(dim=(0, 1)))
        
        # === Compute total physics loss ===
        physics_loss = sum(self._physics_losses.values()) if self._physics_losses else torch.tensor(0.0, device=x.device)
        
        # === Build output dict ===
        result = {
            'output': combined,
            'physics_loss': physics_loss,
            'physics_losses': self._physics_losses,
            'gate_weights': gate_weights,
            'enabled_modules': self.enabled_modules,
            'combination_mode': self.config.combination_mode
        }
        
        if return_components:
            result['components'] = component_outputs
        
        return result
    
    def _forward_compound(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        component_outputs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compound integration: modules feed into each other in a physics-inspired chain.
        
        Chain: Fuller (geometry) → Lloyd (limits) → Morphogenesis (patterns)
               → TPMS (routing) → QbitNexus (quantum) → ParallelUniverse (collapse)
        
        Each module's output becomes the next module's input, creating a
        deep compositional transformation grounded in physics principles.
        
        The idea: 
        1. Fuller establishes geometric structure (tensegrity, stability)
        2. Lloyd constrains computation within physical limits (energy, entropy)
        3. Morphogenesis allows patterns to emerge (Turing, bifurcations)
        4. TPMS routes attention through minimal surfaces (efficiency)
        5. QbitNexus creates quantum superposition states (icosahedral network)
        6. ParallelUniverse explores possibilities and collapses to output
        """
        h = x  # Running hidden state
        residual_scale = 0.1  # Small residual to prevent gradient issues
        
        # Stage 1: Fuller Synergetics - Establish geometric structure
        if self.config.enable_fuller:
            fuller_result = self.fuller(h)
            fuller_out = fuller_result['combined']
            if torch.isnan(fuller_out).any():
                fuller_out = h  # fallback to identity
            h = h + residual_scale * (fuller_out - h)  # Soft update
            component_outputs['fuller'] = fuller_result
            
            if 'tensegrity' in fuller_result:
                tensegrity_loss = fuller_result['tensegrity'].var(dim=-1).mean()
                self._physics_losses['tensegrity'] = tensegrity_loss * self.config.loss_weight_tensegrity
        
        # Stage 2: Lloyd Computational - Apply physical limits to geometry
        if self.config.enable_lloyd:
            lloyd_out, lloyd_info = self.lloyd(h)
            h = h + residual_scale * (lloyd_out - h)
            component_outputs['lloyd'] = {'output': lloyd_out, 'info': lloyd_info}
            
            if 'landauer_cost' in lloyd_info:
                self._physics_losses['lloyd'] = lloyd_info['landauer_cost'] * self.config.loss_weight_lloyd
        
        # Stage 3: Morphogenesis - Patterns emerge from constrained geometry
        if self.config.enable_morphogenesis:
            morpho_result = self.morphogenesis(h)
            morpho_out = morpho_result['combined']
            h = h + residual_scale * (morpho_out - h)
            component_outputs['morphogenesis'] = morpho_result
            
            if 'turing' in morpho_result:
                pattern_var = morpho_result['turing'].var(dim=-1).mean()
                self._physics_losses['turing'] = -torch.log(pattern_var + 1e-6) * self.config.loss_weight_turing
        
        # Stage 4: TPMS Attention - Route through minimal surfaces
        if self.config.enable_tpms:
            tpms_out, tpms_weights = self.tpms(h, mask=attention_mask)
            h = h + residual_scale * (tpms_out - h)
            component_outputs['tpms'] = {'output': tpms_out, 'weights': tpms_weights}
        
        # Stage 5: QbitNexus - Quantum superposition on icosahedral network
        if self.config.enable_qbit_nexus:
            qbit_result = self.qbit_nexus(h)
            qbit_out = qbit_result['output']
            h = h + residual_scale * (qbit_out - h)
            component_outputs['qbit_nexus'] = qbit_result
            
            if 'entanglement' in qbit_result:
                entanglement_magnitude = (qbit_result['entanglement'] ** 2).mean()
                self._physics_losses['entanglement'] = -torch.log(entanglement_magnitude + 1e-6) * self.config.loss_weight_entanglement
        
        # Stage 6: ParallelUniverse - Explore possibilities and collapse
        if self.config.enable_parallel_universe:
            parallel_result = self.parallel_universe(h)
            parallel_out = parallel_result['output']
            # Final stage: stronger integration
            h = 0.5 * h + 0.5 * parallel_out
            component_outputs['parallel_universe'] = parallel_result
            
            if 'interference_pattern' in parallel_result:
                interference_var = parallel_result['interference_pattern'].var()
                self._physics_losses['universe_overlap'] = -torch.log(interference_var + 1e-6) * self.config.loss_weight_universe_overlap
        
        return h

    def get_physics_losses(self) -> Dict[str, torch.Tensor]:
        """Get individual physics loss components"""
        return self._physics_losses.copy()
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get gating statistics from training"""
        if not self._gate_history:
            return {'mean_gates': None, 'module_names': self.enabled_modules}
        
        gate_tensor = torch.stack(self._gate_history)
        mean_gates = gate_tensor.mean(dim=0)
        std_gates = gate_tensor.std(dim=0)
        
        return {
            'mean_gates': mean_gates.tolist(),
            'std_gates': std_gates.tolist(),
            'module_names': self.enabled_modules,
            'num_samples': len(self._gate_history)
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics"""
        self._forward_count = 0
        self._gate_history = []
        self._physics_losses = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics"""
        return {
            'forward_count': self._forward_count,
            'enabled_modules': self.enabled_modules,
            'gate_stats': self.get_gate_statistics(),
            'config': {
                'enable_fuller': self.config.enable_fuller,
                'enable_lloyd': self.config.enable_lloyd,
                'enable_morphogenesis': self.config.enable_morphogenesis,
                'enable_tpms': self.config.enable_tpms,
                'combination_mode': self.config.combination_mode
            }
        }


class QuantumEnhancedHubSync(nn.Module):
    """
    Quantum-enhanced hub synchronization for limb coordination.
    
    Uses quantum coupling dynamics to model inter-limb information flow,
    with TPMS-guided topology for communication channels.
    
    Key features:
    1. Limbs as coupled quantum oscillators (coherent state dynamics)
    2. TPMS channels for bicontinuous information routing
    3. Learnable coupling matrix with physical constraints
    4. Entanglement-based synchronization strength
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_limbs: int = 8,
        coupling_strength: float = 0.1,
        use_tpms_routing: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_limbs = num_limbs
        self.coupling_strength = coupling_strength
        self.use_tpms_routing = use_tpms_routing
        
        # Import quantum coupling
        from sync.quantum_coupling import QuantumCouplingLayer
        
        # Quantum coupling for limb dynamics
        self.quantum_coupling = QuantumCouplingLayer(
            hidden_dim=hidden_dim,
            num_limbs=num_limbs,
            coupling_strength=coupling_strength
        )
        
        # TPMS router for limb communication
        if use_tpms_routing:
            from sync.tpms_attention import TPMSLimbRouter
            self.tpms_router = TPMSLimbRouter(
                hidden_dim=hidden_dim,
                num_limbs=num_limbs,
                tpms_type='gyroid'
            )
        else:
            self.tpms_router = None
        
        # Limb state projections
        self.limb_in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.limb_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Synchronization gate
        self.sync_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Statistics
        self._sync_count = 0
        self._coupling_history: List[torch.Tensor] = []
    
    def forward(
        self,
        limb_states: Dict[str, torch.Tensor],
        hub_state: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Synchronize limb states through quantum coupling.
        
        Args:
            limb_states: Dict of limb_name -> state [batch, hidden]
            hub_state: Central hub state [batch, hidden]
            
        Returns:
            synced_states: Dict of limb_name -> synchronized state
            sync_info: Dict with synchronization statistics
        """
        self._sync_count += 1
        
        # Stack limb states: [batch, num_limbs, hidden]
        limb_names = list(limb_states.keys())
        limb_tensor = torch.stack([limb_states[name] for name in limb_names], dim=1)
        batch = limb_tensor.shape[0]
        
        # Project limb states
        limb_projected = self.limb_in_proj(limb_tensor)  # [batch, num_limbs, hidden]
        
        # Apply quantum coupling dynamics
        coupled, coupling_info = self.quantum_coupling(limb_projected)
        
        # Track coupling matrix
        if self.training and 'coupling_matrix' in coupling_info:
            self._coupling_history.append(coupling_info['coupling_matrix'].detach())
        
        # Apply TPMS routing if enabled
        if self.use_tpms_routing and self.tpms_router is not None:
            routed = self.tpms_router(coupled)
        else:
            routed = coupled
        
        # Compute sync gate for each limb
        # Concatenate hub state with each limb state
        hub_expanded = hub_state.unsqueeze(1).expand(-1, self.num_limbs, -1)
        gate_input = torch.cat([routed, hub_expanded], dim=-1)
        sync_gates = self.sync_gate(gate_input)  # [batch, num_limbs, 1]
        
        # Blend original and coupled states based on gate
        blended = sync_gates * routed + (1 - sync_gates) * limb_projected
        
        # Output projection
        output = self.norm(self.dropout(self.limb_out_proj(blended)) + limb_tensor)
        
        # Unpack back to dict
        synced_states = {}
        for i, name in enumerate(limb_names):
            synced_states[name] = output[:, i, :]
        
        # Compute synchronization metrics
        sync_info = {
            'sync_gates': sync_gates.squeeze(-1),  # [batch, num_limbs]
            'mean_sync_strength': sync_gates.mean().item(),
            'coupling_info': coupling_info,
            'entanglement_entropy': self._compute_entanglement_entropy(coupled)
        }
        
        return synced_states, sync_info
    
    def _compute_entanglement_entropy(self, coupled_states: torch.Tensor) -> float:
        """Compute entanglement entropy between limbs"""
        # Simplified: use correlation matrix eigenvalues
        # [batch, num_limbs, hidden] -> correlation [num_limbs, num_limbs]
        mean_state = coupled_states.mean(dim=0)  # [num_limbs, hidden]
        
        # Normalize
        mean_state_norm = F.normalize(mean_state, dim=-1)
        
        # Correlation matrix
        corr = mean_state_norm @ mean_state_norm.T  # [num_limbs, num_limbs]
        
        # Eigenvalues (treating as density matrix)
        try:
            eigenvalues = torch.linalg.eigvalsh(corr)
        except NotImplementedError:
            # Fallback for MPS: compute on CPU
            corr_cpu = corr.detach().cpu()
            eigenvalues = torch.linalg.eigvalsh(corr_cpu).to(corr.device)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        # Von Neumann entropy
        entropy = -(eigenvalues * torch.log(eigenvalues)).sum()
        
        return entropy.item()
    
    def get_coupling_statistics(self) -> Dict[str, Any]:
        """Get coupling matrix statistics from training"""
        if not self._coupling_history:
            return {'mean_coupling': None}
        
        coupling_tensor = torch.stack(self._coupling_history)
        mean_coupling = coupling_tensor.mean(dim=0)
        
        return {
            'mean_coupling': mean_coupling,
            'num_samples': len(self._coupling_history)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        return {
            'sync_count': self._sync_count,
            'coupling_stats': self.get_coupling_statistics(),
            'use_tpms_routing': self.use_tpms_routing
        }


# === Factory function for easy creation ===

def create_geometric_physics_layer(
    hidden_dim: int,
    preset: str = 'full',
    num_heads: int = 8,
    **kwargs
) -> GeometricPhysicsLayer:
    """
    Factory function to create GeometricPhysicsLayer with presets.
    
    Args:
        hidden_dim: Hidden dimension
        preset: One of 'full', 'lightweight', 'geometry_only', 'physics_only'
        num_heads: Number of attention heads
        **kwargs: Override specific config options
        
    Returns:
        Configured GeometricPhysicsLayer
    """
    presets = {
        'full': GeometricPhysicsConfig(
            enable_fuller=True,
            enable_lloyd=True,
            enable_morphogenesis=True,
            enable_tpms=True,
            enable_qbit_nexus=True,
            enable_parallel_universe=True
        ),
        'lightweight': GeometricPhysicsConfig(
            enable_fuller=True,
            enable_lloyd=False,
            enable_morphogenesis=False,
            enable_tpms=True,
            enable_qbit_nexus=False,
            enable_parallel_universe=False,
            fuller_geodesic_frequency=1,
            morpho_diffusion_steps=1
        ),
        'geometry_only': GeometricPhysicsConfig(
            enable_fuller=True,
            enable_lloyd=False,
            enable_morphogenesis=True,
            enable_tpms=True,
            enable_qbit_nexus=True,
            enable_parallel_universe=False
        ),
        'physics_only': GeometricPhysicsConfig(
            enable_fuller=False,
            enable_lloyd=True,
            enable_morphogenesis=True,
            enable_tpms=False,
            enable_qbit_nexus=False,
            enable_parallel_universe=True
        ),
        'quantum': GeometricPhysicsConfig(
            # Quantum-focused preset: QbitNexus + ParallelUniverse + Lloyd
            enable_fuller=False,
            enable_lloyd=True,
            enable_morphogenesis=False,
            enable_tpms=False,
            enable_qbit_nexus=True,
            enable_parallel_universe=True
        ),
        'compound': GeometricPhysicsConfig(
            # COMPOUND INTEGRATION: All modules feed into each other sequentially
            # Physics chain: geometry → limits → patterns → routing → quantum → collapse
            enable_fuller=True,
            enable_lloyd=True,
            enable_morphogenesis=True,
            enable_tpms=True,
            enable_qbit_nexus=True,
            enable_parallel_universe=True,
            combination_mode='compound'  # Sequential chained processing
        )
    }
    
    config = presets.get(preset, presets['full'])
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return GeometricPhysicsLayer(
        hidden_dim=hidden_dim,
        config=config,
        num_heads=num_heads
    )


# === Test ===

if __name__ == "__main__":
    print("Testing GeometricPhysicsLayer...")
    
    # Create layer with full config
    layer = create_geometric_physics_layer(
        hidden_dim=256,
        preset='full',
        num_heads=8
    )
    
    print(f"Enabled modules: {layer.enabled_modules}")
    
    # Test forward pass
    batch, seq_len = 2, 16
    x = torch.randn(batch, seq_len, 256)
    
    result = layer(x, return_components=True)
    
    print(f"\nOutput shape: {result['output'].shape}")
    print(f"Physics loss: {result['physics_loss'].item():.6f}")
    print(f"Gate weights shape: {result['gate_weights'].shape}")
    print(f"Physics losses: {list(result['physics_losses'].keys())}")
    
    # Test backward pass
    loss = result['output'].mean() + result['physics_loss']
    loss.backward()
    print(f"\nBackward pass successful")
    
    # Test quantum-enhanced hub sync
    print("\n" + "="*50)
    print("Testing QuantumEnhancedHubSync...")
    
    hub_sync = QuantumEnhancedHubSync(
        hidden_dim=256,
        num_limbs=8,
        coupling_strength=0.1
    )
    
    # Create mock limb states
    limb_states = {
        'perception': torch.randn(2, 256),
        'memory': torch.randn(2, 256),
        'planning': torch.randn(2, 256),
        'language': torch.randn(2, 256),
        'spatial': torch.randn(2, 256),
        'reasoning': torch.randn(2, 256),
        'metacognition': torch.randn(2, 256),
        'action': torch.randn(2, 256)
    }
    hub_state = torch.randn(2, 256)
    
    synced_states, sync_info = hub_sync(limb_states, hub_state)
    
    print(f"Synced states: {list(synced_states.keys())}")
    print(f"Mean sync strength: {sync_info['mean_sync_strength']:.4f}")
    print(f"Entanglement entropy: {sync_info['entanglement_entropy']:.4f}")
    
    # Test presets
    print("\n" + "="*50)
    print("Testing presets...")
    
    for preset in ['full', 'lightweight', 'geometry_only', 'physics_only']:
        layer = create_geometric_physics_layer(256, preset=preset)
        print(f"  {preset}: {layer.enabled_modules}")
    
    print("\nAll GeometricPhysicsLayer tests passed!")
