"""
Quantum Coupling Layer for OctoTetrahedral AGI

Implements learnable coupling matrix between 8 limbs based on
quantum harmonic oscillator dynamics.

Physics mapping:
- Each limb = quantum oscillator with frequency ω_i
- Coupling g_ij determines information exchange rate
- Zero-point energy (+7) ensures non-zero baseline activation
- Coherent state regularization encourages Gaussian hidden states

Usage:
    coupling = QuantumCouplingLayer(hidden_dim=256, num_limbs=8)
    enhanced_states, coupling_loss = coupling(limb_states)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class QuantumCouplingLayer(nn.Module):
    """
    Learnable coupling matrix for 8 limbs modeled as coupled quantum oscillators.
    
    Hamiltonian:
        H = Σᵢ ħωᵢ(aᵢ†aᵢ + ½) + Σᵢⱼ gᵢⱼ(aᵢ†aⱼ + aᵢaⱼ†)
    
    Neural mapping:
        - ωᵢ (frequency) → layer processing rate
        - gᵢⱼ (coupling) → cross-attention / residual strength
        - aᵢ†aᵢ (occupation) → ||hidden_state||²
    """
    
    LIMB_NAMES = [
        'perception', 'memory', 'planning', 'language',
        'spatial', 'reasoning', 'metacognition', 'action'
    ]
    
    def __init__(
        self,
        hidden_dim: int,
        num_limbs: int = 8,
        zero_point_energy: float = 7.0,
        coupling_strength: float = 0.1,
        enable_rabi_oscillation: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_limbs = num_limbs
        self.enable_rabi = enable_rabi_oscillation
        
        # Zero-point energy (the +7 in z = z3 + 7)
        self.zero_point = nn.Parameter(torch.tensor(zero_point_energy))
        
        # Natural frequencies ωᵢ for each limb (learnable)
        # Initialized with different base frequencies
        omega_init = torch.ones(num_limbs) + torch.arange(num_limbs).float() * 0.1
        self.omega = nn.Parameter(omega_init)
        
        # Coupling matrix gᵢⱼ (symmetric, zero diagonal)
        # This is the key learnable parameter for quantum-like dynamics
        coupling_init = torch.randn(num_limbs, num_limbs) * coupling_strength
        # Make symmetric
        coupling_init = (coupling_init + coupling_init.T) / 2
        # Zero diagonal (no self-coupling)
        coupling_init.fill_diagonal_(0)
        self.coupling = nn.Parameter(coupling_init)
        
        # Projection for coupling interaction
        self.coupling_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Energy level projections (maps hidden state to "occupation number")
        self.occupation_proj = nn.Linear(hidden_dim, 1)
        
        # Gate for blending coupled vs original states
        self.blend_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Statistics tracking
        self._coupling_energy = 0.0
        self._total_energy = 0.0
        self._rabi_amplitude = 0.0
    
    def get_coupling_matrix(self) -> torch.Tensor:
        """
        Return the symmetric coupling matrix with zero diagonal.
        """
        # Enforce symmetry and zero diagonal
        coupling = (self.coupling + self.coupling.T) / 2
        mask = torch.ones_like(coupling) - torch.eye(self.num_limbs, device=coupling.device)
        return coupling * mask
    
    def compute_hamiltonian_energy(
        self,
        limb_states: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute system energy from limb states.
        
        E = Σᵢ ħωᵢ(nᵢ + ½) + Σᵢⱼ gᵢⱼ⟨ψᵢ|ψⱼ⟩
        
        Args:
            limb_states: Dict of limb_name -> hidden_state [batch, seq, hidden]
        
        Returns:
            individual_energies: [batch, num_limbs]
            coupling_energy: [batch]
        """
        batch_size = list(limb_states.values())[0].shape[0]
        device = list(limb_states.values())[0].device
        
        # Get ordered states
        states = []
        for name in self.LIMB_NAMES:
            if name in limb_states:
                states.append(limb_states[name])
            else:
                # Placeholder if limb not present
                states.append(torch.zeros(batch_size, 1, self.hidden_dim, device=device))
        
        # Stack: [batch, num_limbs, seq, hidden]
        # Use mean over sequence for energy calculation
        stacked = torch.stack([s.mean(dim=1) for s in states], dim=1)  # [batch, num_limbs, hidden]
        
        # Occupation numbers: ||ψᵢ||² (proxy for quantum number n)
        occupation = (stacked ** 2).sum(dim=-1)  # [batch, num_limbs]
        
        # Individual limb energies: ħωᵢ(nᵢ + ½)
        hbar = 1.0  # Normalized
        individual_energy = hbar * self.omega.unsqueeze(0) * (occupation + 0.5)
        
        # Add zero-point offset
        individual_energy = individual_energy + self.zero_point
        
        # Coupling energy: gᵢⱼ⟨ψᵢ|ψⱼ⟩
        coupling_matrix = self.get_coupling_matrix()  # [num_limbs, num_limbs]
        
        # Inner products between limb states
        # stacked: [batch, num_limbs, hidden]
        inner_products = torch.bmm(
            stacked, 
            stacked.transpose(1, 2)
        )  # [batch, num_limbs, num_limbs]
        
        # Weighted sum of inner products
        coupling_energy = (coupling_matrix.unsqueeze(0) * inner_products).sum(dim=(1, 2))  # [batch]
        
        return individual_energy, coupling_energy
    
    def apply_coupling(
        self,
        limb_states: Dict[str, torch.Tensor],
        time_step: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Apply quantum coupling dynamics to limb states.
        
        This simulates information exchange between limbs based on coupling strengths.
        
        Args:
            limb_states: Dict of limb_name -> hidden_state [batch, seq, hidden]
            time_step: Evolution time (affects Rabi oscillation)
            
        Returns:
            Dict of limb_name -> coupled_state [batch, seq, hidden]
        """
        device = list(limb_states.values())[0].device
        batch_size = list(limb_states.values())[0].shape[0]
        
        coupling_matrix = self.get_coupling_matrix().to(device)
        
        # Collect states in order
        ordered_states = []
        for name in self.LIMB_NAMES:
            if name in limb_states:
                ordered_states.append(limb_states[name])
            else:
                seq_len = list(limb_states.values())[0].shape[1]
                ordered_states.append(torch.zeros(batch_size, seq_len, self.hidden_dim, device=device))
        
        # Stack: [batch, seq, num_limbs, hidden]
        stacked = torch.stack(ordered_states, dim=2)
        
        # Project for coupling
        projected = self.coupling_proj(stacked)  # [batch, seq, num_limbs, hidden]
        
        # Apply coupling matrix: each limb receives weighted sum from others
        # coupling_matrix: [num_limbs, num_limbs]
        # projected: [batch, seq, num_limbs, hidden]
        coupled = torch.einsum('ij,bsjd->bsid', coupling_matrix, projected)
        
        # Rabi oscillation modulation (optional)
        if self.enable_rabi:
            # Rabi frequency Ω = 2g
            rabi_freq = 2 * coupling_matrix.abs()
            
            # Oscillation amplitude sin²(Ωt/2)
            # This creates oscillating exchange between adjacent limbs
            phase = rabi_freq * time_step / 2
            rabi_amplitude = torch.sin(phase) ** 2
            
            # Modulate coupling by Rabi amplitude
            # For adjacent limbs (circular topology)
            for i in range(self.num_limbs):
                j = (i + 1) % self.num_limbs
                amp = rabi_amplitude[i, j]
                
                # Exchange some activation between i and j
                exchange = amp * (projected[:, :, j, :] - projected[:, :, i, :])
                coupled[:, :, i, :] = coupled[:, :, i, :] + 0.1 * exchange
                coupled[:, :, j, :] = coupled[:, :, j, :] - 0.1 * exchange
            
            self._rabi_amplitude = rabi_amplitude.mean().item()
        
        # Blend coupled with original using learned gate
        output_states = {}
        for idx, name in enumerate(self.LIMB_NAMES):
            if name in limb_states:
                original = limb_states[name]
                coupled_state = coupled[:, :, idx, :]
                
                # Gate: how much to blend
                gate_input = torch.cat([original, coupled_state], dim=-1)
                gate = self.blend_gate(gate_input)
                
                # Blend and normalize
                blended = gate * coupled_state + (1 - gate) * original
                output_states[name] = self.norm(blended)
        
        return output_states
    
    def coherent_state_loss(
        self,
        limb_states: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss encouraging coherent (Gaussian) state distributions.
        
        Coherent states |α⟩ have minimum uncertainty and Gaussian distributions.
        We encourage this by penalizing deviation from Gaussian statistics.
        
        Args:
            limb_states: Dict of limb_name -> hidden_state
            
        Returns:
            Scalar loss encouraging Gaussian distributions
        """
        losses = []
        
        for name, state in limb_states.items():
            # Flatten to [batch * seq, hidden]
            flat = state.reshape(-1, self.hidden_dim)
            
            # Mean and std per dimension
            mean = flat.mean(dim=0)
            std = flat.std(dim=0)
            
            # For Gaussian: skewness should be ~0, kurtosis should be ~3
            # Compute sample skewness and kurtosis
            centered = flat - mean.unsqueeze(0)
            
            # Skewness: E[(x-μ)³] / σ³
            skewness = (centered ** 3).mean(dim=0) / (std ** 3 + 1e-8)
            skew_loss = (skewness ** 2).mean()
            
            # Excess kurtosis: E[(x-μ)⁴] / σ⁴ - 3
            kurtosis = (centered ** 4).mean(dim=0) / (std ** 4 + 1e-8) - 3
            kurt_loss = (kurtosis ** 2).mean()
            
            losses.append(skew_loss + kurt_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)
    
    def quantization_loss(
        self,
        limb_states: Dict[str, torch.Tensor],
        num_levels: int = 8
    ) -> torch.Tensor:
        """
        Compute loss encouraging quantized (discrete) activation levels.
        
        This helps create the discrete energy levels E_n = ħω(n + ½).
        
        Args:
            limb_states: Dict of limb_name -> hidden_state
            num_levels: Target number of quantization levels
            
        Returns:
            Scalar loss encouraging discrete levels
        """
        losses = []
        
        for name, state in limb_states.items():
            # Use activation magnitudes
            magnitudes = state.norm(dim=-1)  # [batch, seq]
            
            # Define target quantization levels
            min_val = magnitudes.min()
            max_val = magnitudes.max()
            levels = torch.linspace(min_val, max_val, num_levels, device=state.device)
            
            # Distance to nearest level
            magnitudes_flat = magnitudes.reshape(-1, 1)  # [batch*seq, 1]
            distances = (magnitudes_flat - levels.unsqueeze(0)).abs()  # [batch*seq, num_levels]
            min_distances = distances.min(dim=1)[0]  # [batch*seq]
            
            # Penalize being far from any level
            quant_loss = min_distances.mean()
            losses.append(quant_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)
    
    def forward(
        self,
        limb_states: Dict[str, torch.Tensor],
        time_step: float = 1.0,
        return_losses: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass: apply quantum coupling and compute auxiliary losses.
        
        Args:
            limb_states: Dict of limb_name -> hidden_state [batch, seq, hidden]
            time_step: Evolution time for Rabi oscillation
            return_losses: Whether to compute regularization losses
            
        Returns:
            coupled_states: Dict of limb_name -> coupled_state
            losses: Dict of loss_name -> loss_value (if return_losses=True)
        """
        # Apply coupling dynamics
        coupled_states = self.apply_coupling(limb_states, time_step)
        
        losses = {}
        if return_losses:
            # Compute Hamiltonian energy (for monitoring)
            individual_E, coupling_E = self.compute_hamiltonian_energy(coupled_states)
            self._coupling_energy = coupling_E.mean().item()
            self._total_energy = individual_E.sum(dim=1).mean().item() + self._coupling_energy
            
            # Coherent state loss (encourage Gaussian distributions)
            losses['coherent_loss'] = self.coherent_state_loss(coupled_states) * 0.01
            
            # Quantization loss (encourage discrete levels)
            losses['quantization_loss'] = self.quantization_loss(coupled_states) * 0.001
            
            # Energy regularization (prevent runaway energy)
            losses['energy_reg'] = (individual_E.sum(dim=1).mean() - self.num_limbs * self.zero_point).abs() * 0.0001
        
        return coupled_states, losses
    
    def get_stats(self) -> Dict[str, float]:
        """Get coupling statistics"""
        coupling_matrix = self.get_coupling_matrix()
        
        return {
            'zero_point_energy': self.zero_point.item(),
            'mean_omega': self.omega.mean().item(),
            'mean_coupling': coupling_matrix.abs().mean().item(),
            'max_coupling': coupling_matrix.abs().max().item(),
            'coupling_energy': self._coupling_energy,
            'total_energy': self._total_energy,
            'rabi_amplitude': self._rabi_amplitude
        }


class QuantumEnhancedHubSync:
    """
    Enhanced hub synchronization with quantum coupling dynamics.
    
    Wraps the quantum coupling layer and integrates with HubSync.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_limbs: int = 8,
        zero_point_energy: float = 7.0,
        coupling_strength: float = 0.1
    ):
        self.coupling_layer = QuantumCouplingLayer(
            hidden_dim=hidden_dim,
            num_limbs=num_limbs,
            zero_point_energy=zero_point_energy,
            coupling_strength=coupling_strength
        )
        
        # Time tracking for Rabi oscillation
        self._step = 0
    
    def process_limb_outputs(
        self,
        limb_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process limb outputs through quantum coupling.
        """
        # Time evolves with steps (normalized)
        time_step = (self._step % 100) / 100.0 * 2 * math.pi
        
        coupled, losses = self.coupling_layer(limb_outputs, time_step)
        
        self._step += 1
        
        return coupled, losses
    
    def get_coupling_matrix(self) -> torch.Tensor:
        """Return the current coupling matrix for visualization"""
        return self.coupling_layer.get_coupling_matrix()
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics"""
        return self.coupling_layer.get_stats()


if __name__ == "__main__":
    print("Testing QuantumCouplingLayer...")
    
    # Create layer
    hidden_dim = 256
    coupling = QuantumCouplingLayer(hidden_dim=hidden_dim, num_limbs=8)
    
    # Create mock limb states
    batch_size = 4
    seq_len = 32
    
    limb_states = {}
    for name in QuantumCouplingLayer.LIMB_NAMES:
        limb_states[name] = torch.randn(batch_size, seq_len, hidden_dim)
    
    print(f"Input limb states: {len(limb_states)} limbs")
    
    # Forward pass
    coupled_states, losses = coupling(limb_states, time_step=1.0)
    
    print(f"\nOutput coupled states: {len(coupled_states)} limbs")
    for name, state in coupled_states.items():
        print(f"  {name}: shape {state.shape}")
    
    print(f"\nLosses:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.6f}")
    
    print(f"\nCoupling matrix:")
    coupling_matrix = coupling.get_coupling_matrix()
    print(coupling_matrix)
    
    print(f"\nStats:")
    stats = coupling.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nQuantumCouplingLayer tests passed!")
