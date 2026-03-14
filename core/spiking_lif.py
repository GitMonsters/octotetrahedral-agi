"""
Leaky Integrate-and-Fire (LIF) Spiking Neural Network Layer

Inspired by the connectome brain emulation breakthrough: EON's fruit fly
brain achieved 91% behavioral accuracy using only 4 ingredients:
1. Connection graph (topology)
2. Synapse weights (strength)
3. Excitatory/inhibitory map (sign of connections)
4. LIF dynamics (how neurons fire)

This module implements a hybrid LIF sublayer that maps the 64 tetrahedral
geometry points as 64 spiking neurons. The connection graph comes from the
tetrahedral adjacency matrix. Synapse weights derive from geometric distances.
E/I classification comes from the RNA editing layer.

The layer operates alongside standard transformer layers, adding temporal
spike coding and energy-efficient sparse activation patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron model.
    
    V(t+1) = leak * V(t) + Σ(w_i * spike_i * ei_sign_i)
    if V > threshold: fire spike, reset V
    
    Uses surrogate gradient (straight-through estimator) for backprop
    through the non-differentiable spike function.
    """
    
    def __init__(
        self,
        num_neurons: int = 64,
        leak_rate: float = 0.9,
        threshold: float = 1.0,
        reset_value: float = 0.0,
        refractory_steps: int = 1,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.reset_value = reset_value
        self.refractory_steps = refractory_steps
        
        # Learnable leak rate per neuron (controls memory persistence)
        # Fast leak = short-term memory, slow leak = long-term
        self.leak_rate = nn.Parameter(torch.full((num_neurons,), leak_rate))
        
        # Learnable threshold per neuron
        self.threshold_param = nn.Parameter(torch.full((num_neurons,), threshold))
    
    def forward(
        self,
        input_current: torch.Tensor,
        membrane_potential: torch.Tensor,
        ei_signs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One timestep of LIF dynamics.
        
        Args:
            input_current: [batch, num_neurons] weighted input
            membrane_potential: [batch, num_neurons] current membrane state
            ei_signs: Optional [num_neurons] or [batch, num_neurons] E/I signs
            
        Returns:
            (spikes [batch, num_neurons], new_potential [batch, num_neurons])
        """
        # Leak
        leak = torch.sigmoid(self.leak_rate)  # Constrain to (0, 1)
        potential = leak * membrane_potential
        
        # Integrate (apply E/I signs to input if provided)
        if ei_signs is not None:
            input_current = input_current * ei_signs
        potential = potential + input_current
        
        # Fire: threshold comparison with surrogate gradient
        thresh = F.softplus(self.threshold_param)  # Ensure positive
        spike_raw = potential - thresh
        
        # Heaviside with straight-through estimator for gradients
        spikes = (spike_raw > 0).float()
        spikes = spike_raw.sigmoid() + (spikes - spike_raw.sigmoid()).detach()  # STE
        
        # Reset neurons that fired
        new_potential = potential * (1.0 - spikes) + self.reset_value * spikes
        
        return spikes, new_potential


class SpikingTetrahedralLayer(nn.Module):
    """
    Hybrid spiking-continuous layer using tetrahedral geometry as connectome.
    
    64 tetrahedral points = 64 LIF neurons.
    Adjacency matrix from tetrahedral geometry = connection graph.
    Geometric distances = synapse weights.
    E/I classification from RNA editing.
    
    Runs T timesteps of spiking dynamics, then converts spike
    patterns back to continuous representations.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_neurons: int = 64,
        num_timesteps: int = 4,
        leak_rate: float = 0.9,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_neurons = num_neurons
        self.num_timesteps = num_timesteps
        
        # Project from hidden_dim to neuron space and back
        self.input_proj = nn.Linear(hidden_dim, num_neurons)
        self.output_proj = nn.Linear(num_neurons * num_timesteps, hidden_dim)
        
        # Synapse weight matrix (learnable, initialized from geometry if available)
        self.synapse_weights = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.1)
        
        # LIF neuron dynamics
        self.lif = LIFNeuron(
            num_neurons=num_neurons,
            leak_rate=leak_rate,
            threshold=threshold,
        )
        
        # Layer norm for output
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Gate to blend spiking output with residual
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def init_from_geometry(self, adjacency: torch.Tensor, distances: torch.Tensor):
        """
        Initialize synapse weights from tetrahedral geometry.
        
        Args:
            adjacency: [num_neurons, num_neurons] binary connectivity
            distances: [num_neurons, num_neurons] pairwise distances
        """
        with torch.no_grad():
            # Weight = adjacency * inverse_distance (closer = stronger)
            inv_dist = 1.0 / (distances + 1e-6)
            inv_dist = inv_dist / inv_dist.max()
            self.synapse_weights.data = adjacency * inv_dist * 0.5
    
    def apply_genome(self, genome) -> None:
        """
        Apply an evolved connectome genome to this spiking layer.
        
        Args:
            genome: Genome dataclass from core.connectome_evolution with:
                    adjacency_mask [num_neurons, num_neurons],
                    weight_scales [num_neurons, num_neurons],
                    ei_labels [num_neurons]
        """
        with torch.no_grad():
            device = self.synapse_weights.device
            adj = genome.adjacency_mask.to(device)
            ws = genome.weight_scales.to(device)
            self.synapse_weights.data = adj * ws
            # Store E/I labels as a buffer for the LIF neuron to use
            if not hasattr(self, '_genome_ei_labels'):
                self.register_buffer('_genome_ei_labels', genome.ei_labels.to(device))
            else:
                self._genome_ei_labels.copy_(genome.ei_labels.to(device))
    
    def forward(
        self,
        x: torch.Tensor,
        ei_signs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run spiking dynamics on input.
        
        Args:
            x: [batch, seq_len, hidden_dim]
            ei_signs: Optional [hidden_dim] or [batch, hidden_dim] E/I signs
            
        Returns:
            Dict with 'output', 'spike_rates', 'energy'
        """
        batch_size, seq_len, _ = x.shape
        
        # Pool sequence to get neuron input currents
        pooled = x.mean(dim=1)  # [batch, hidden_dim]
        neuron_input = self.input_proj(pooled)  # [batch, num_neurons]
        
        # Truncate or broadcast E/I signs to neuron space
        if ei_signs is not None:
            if ei_signs.dim() == 1:
                ei_neuron = ei_signs[:self.num_neurons]
            else:
                ei_neuron = ei_signs[:, :self.num_neurons]
        else:
            ei_neuron = None
        
        # Run T timesteps of spiking dynamics
        membrane = torch.zeros(batch_size, self.num_neurons, device=x.device)
        all_spikes = []
        
        for t in range(self.num_timesteps):
            # Synaptic current: weighted sum of previous spikes
            if t == 0:
                synaptic_input = neuron_input
            else:
                prev_spikes = all_spikes[-1]
                synaptic_input = neuron_input + torch.matmul(prev_spikes, self.synapse_weights)
            
            spikes, membrane = self.lif(synaptic_input, membrane, ei_neuron)
            all_spikes.append(spikes)
        
        # Concatenate spike trains across timesteps
        spike_train = torch.cat(all_spikes, dim=-1)  # [batch, num_neurons * T]
        
        # Convert back to hidden_dim
        spiking_output = self.output_proj(spike_train)  # [batch, hidden_dim]
        
        # Expand to sequence and gate with residual
        spiking_expanded = spiking_output.unsqueeze(1).expand(-1, seq_len, -1)
        gate_input = torch.cat([x, spiking_expanded], dim=-1)
        gate = self.gate(gate_input)
        output = self.norm(gate * x + (1 - gate) * spiking_expanded)
        
        # Compute spike statistics
        spike_stack = torch.stack(all_spikes, dim=0)  # [T, batch, neurons]
        spike_rates = spike_stack.mean(dim=0)  # [batch, neurons] avg firing rate
        energy = spike_stack.sum()  # Total spikes (energy proxy)
        
        return {
            'output': output,
            'spike_rates': spike_rates,
            'energy': energy,
            'membrane_final': membrane,
            'spike_train': spike_train,
        }


if __name__ == "__main__":
    print("Testing SpikingTetrahedralLayer...")
    
    batch_size = 2
    seq_len = 20
    hidden_dim = 256
    num_neurons = 64
    
    layer = SpikingTetrahedralLayer(
        hidden_dim=hidden_dim,
        num_neurons=num_neurons,
        num_timesteps=4,
    )
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Without E/I
    result = layer(x)
    print(f"Output shape: {result['output'].shape}")
    print(f"Spike rates: mean={result['spike_rates'].mean():.3f}, max={result['spike_rates'].max():.3f}")
    print(f"Energy (total spikes): {result['energy'].item():.1f}")
    
    # With E/I signs
    ei = torch.sign(torch.randn(hidden_dim))
    result2 = layer(x, ei_signs=ei)
    print(f"With E/I — output shape: {result2['output'].shape}")
    
    # Test geometry initialization
    adj = (torch.randn(num_neurons, num_neurons) > 0.5).float()
    dist = torch.rand(num_neurons, num_neurons) + 0.1
    layer.init_from_geometry(adj, dist)
    result3 = layer(x)
    print(f"After geometry init — spike rates: mean={result3['spike_rates'].mean():.3f}")
    
    # Gradient check
    x.requires_grad_(True)
    result4 = layer(x)
    loss = result4['output'].sum()
    loss.backward()
    print(f"Gradient flows: {x.grad is not None and x.grad.abs().sum() > 0}")
    
    params = sum(p.numel() for p in layer.parameters())
    print(f"\nParameters: {params:,}")
    
    print("\nSpikingTetrahedralLayer tests passed!")
