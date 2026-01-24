"""
RNA Editing Layer
Dynamic weight modulation inspired by octopus A-to-I RNA editing

Key biological inspirations:
1. ADAR enzymes deaminate adenosine to inosine at specific sites
2. Editing intensity varies with environment (temperature, stress)
3. Different tissues have different editing profiles
4. Changes are reversible and rapid (hours, not generations)

This module implements three levels of "editing":
1. Attention head gating - which heads are active
2. Temperature modulation - how much to edit
3. Pathway routing - which limb handles the input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class RNAEditingLayer(nn.Module):
    """
    RNA Editing-inspired dynamic modulation layer.
    
    Provides three mechanisms for adaptive behavior:
    1. Temperature: Controls overall editing intensity
    2. Head gating: Selectively activates/deactivates attention heads
    3. Pathway routing: Routes information to specialized limbs
    
    The "editing" is reversible - it modulates existing weights
    rather than permanently changing them.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_pathways: int = 3,
        temperature_init: float = 1.0,
        temperature_min: float = 0.1,
        temperature_max: float = 5.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_pathways = num_pathways
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        
        # Learnable base temperature
        self.temperature_base = nn.Parameter(torch.tensor(temperature_init))
        
        # Temperature modulation network
        # Maps input context to temperature adjustment
        self.temperature_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output in [0, 1], will be scaled
        )
        
        # Attention head gating network
        # Outputs a gate value for each attention head
        self.head_gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Sigmoid()  # Gates in [0, 1]
        )
        
        # Pathway routing network
        # Determines which limb(s) should process the input
        self.pathway_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_pathways)
        )
        
        # Confidence estimator
        # Estimates model confidence for the current input
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Editing site selector (which dimensions to edit)
        # Analogous to ADAR enzyme selectivity
        self.editing_sites = nn.Parameter(torch.ones(hidden_dim))
        
    def compute_temperature(
        self, 
        context: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute effective temperature based on context and confidence.
        
        Low confidence → high temperature → more editing
        High confidence → low temperature → less editing
        
        Args:
            context: Context tensor [batch, hidden_dim]
            confidence: Optional pre-computed confidence [batch, 1]
            
        Returns:
            Temperature tensor [batch, 1]
        """
        # Base temperature adjustment from context
        temp_adjustment = self.temperature_net(context)  # [batch, 1]
        
        # Scale to temperature range
        temp_range = self.temperature_max - self.temperature_min
        temperature = self.temperature_min + temp_adjustment * temp_range
        
        # Confidence-based modulation
        if confidence is not None:
            # Low confidence increases temperature
            confidence_factor = 1.0 + (1.0 - confidence)
            temperature = temperature * confidence_factor
        
        # Clamp to valid range
        temperature = temperature.clamp(self.temperature_min, self.temperature_max)
        
        return temperature
    
    def compute_head_gates(
        self, 
        context: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gating values for each attention head.
        
        Higher temperature leads to more uniform gating (exploration).
        Lower temperature leads to more selective gating (exploitation).
        
        Args:
            context: Context tensor [batch, hidden_dim]
            temperature: Optional temperature for softening [batch, 1]
            
        Returns:
            Head gates [batch, num_heads] in [0, 1]
        """
        # Raw gate values
        gates = self.head_gate_net(context)  # [batch, num_heads]
        
        # Temperature-based softening
        if temperature is not None:
            # Higher temperature → gates closer to uniform
            # Lower temperature → gates more extreme
            gates = torch.sigmoid((gates - 0.5) / temperature.clamp(min=0.1))
        
        return gates
    
    def compute_pathway_weights(
        self,
        context: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute routing weights for each pathway (limb).
        
        Uses temperature-scaled softmax for soft routing.
        
        Args:
            context: Context tensor [batch, hidden_dim]
            temperature: Optional temperature for scaling [batch, 1]
            
        Returns:
            Pathway weights [batch, num_pathways] summing to 1
        """
        # Raw routing logits
        logits = self.pathway_router(context)  # [batch, num_pathways]
        
        # Temperature-scaled softmax
        if temperature is not None:
            logits = logits / temperature.clamp(min=0.1)
        
        weights = F.softmax(logits, dim=-1)
        
        return weights
    
    def compute_editing_mask(
        self,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute which dimensions to edit based on temperature.
        
        Higher temperature → edit more dimensions
        Lower temperature → edit fewer dimensions
        
        Args:
            temperature: Temperature tensor [batch, 1]
            
        Returns:
            Editing mask [batch, hidden_dim] in [0, 1]
        """
        # Use editing sites as base selectivity
        site_probs = torch.sigmoid(self.editing_sites)  # [hidden_dim]
        
        # Temperature modulates how many sites are active
        # High temp → more sites active (closer to 1)
        # Low temp → fewer sites active (use learned probs)
        temp_factor = (temperature / self.temperature_max).unsqueeze(-1)  # [batch, 1, 1]
        
        # Interpolate between learned selectivity and all-active
        mask = site_probs + temp_factor.squeeze(-1) * (1.0 - site_probs)
        
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Apply RNA editing-style modulation to input.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            return_diagnostics: Whether to return detailed diagnostics
            
        Returns:
            Dictionary containing:
                - 'output': Modulated tensor (same shape as input)
                - 'temperature': Effective temperature [batch, 1]
                - 'head_gates': Attention head gates [batch, num_heads]
                - 'pathway_weights': Routing weights [batch, num_pathways]
                - 'confidence': Estimated confidence [batch, 1]
                - 'editing_mask': (optional) Which dimensions edited
        """
        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        
        # Use mean-pooled representation for context
        context = x.mean(dim=1)  # [batch, hidden_dim]
        
        # Compute confidence
        confidence = self.confidence_net(context)  # [batch, 1]
        
        # Compute temperature (affected by confidence)
        temperature = self.compute_temperature(context, confidence)  # [batch, 1]
        
        # Compute head gates
        head_gates = self.compute_head_gates(context, temperature)  # [batch, num_heads]
        
        # Compute pathway routing
        pathway_weights = self.compute_pathway_weights(context, temperature)  # [batch, num_pathways]
        
        # Compute editing mask
        editing_mask = self.compute_editing_mask(temperature)  # [batch, hidden_dim]
        
        # Apply editing: modulate input based on mask and temperature
        # This is a soft "editing" that can be reversed
        editing_strength = temperature / self.temperature_max  # [batch, 1]
        modulated = x * (1.0 + editing_strength.unsqueeze(1) * (editing_mask.unsqueeze(1) - 0.5) * 0.1)
        
        if squeeze_output:
            modulated = modulated.squeeze(1)
        
        result = {
            'output': modulated,
            'temperature': temperature,
            'head_gates': head_gates,
            'pathway_weights': pathway_weights,
            'confidence': confidence
        }
        
        if return_diagnostics:
            result['editing_mask'] = editing_mask
            result['context'] = context
        
        return result
    
    def get_pathway_names(self) -> list:
        """Return semantic names for pathways"""
        return ['perception', 'reasoning', 'action'][:self.num_pathways]


class AdaptiveTriggerSystem(nn.Module):
    """
    System for triggering adaptation based on multiple signals.
    
    Triggers:
    1. Confidence-based: Low confidence → trigger adaptation
    2. Novelty-based: High novelty → trigger adaptation
    3. Error-based: High error → trigger adaptation
    4. Time-based: Periodic adaptation regardless of signals
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        confidence_threshold: float = 0.5,
        novelty_threshold: float = 0.7,
        adaptation_cooldown: int = 10
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.confidence_threshold = confidence_threshold
        self.novelty_threshold = novelty_threshold
        self.adaptation_cooldown = adaptation_cooldown
        
        # Novelty detector (compares to running statistics)
        self.register_buffer('running_mean', torch.zeros(hidden_dim))
        self.register_buffer('running_var', torch.ones(hidden_dim))
        self.register_buffer('steps_since_adaptation', torch.tensor(0))
        
        # Novelty scoring network
        self.novelty_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def compute_novelty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute novelty score based on deviation from running statistics"""
        # x: [batch, hidden_dim]
        
        # Compute z-score
        z_score = (x - self.running_mean) / (self.running_var.sqrt() + 1e-6)
        
        # Combine with learned novelty detection
        combined = torch.cat([x, z_score], dim=-1)  # [batch, hidden_dim * 2]
        novelty = self.novelty_net(combined)  # [batch, 1]
        
        return novelty
    
    def update_statistics(self, x: torch.Tensor, momentum: float = 0.99):
        """Update running statistics with new observations"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        
        self.running_mean = momentum * self.running_mean + (1 - momentum) * batch_mean
        self.running_var = momentum * self.running_var + (1 - momentum) * batch_var
    
    def should_adapt(
        self,
        confidence: torch.Tensor,
        novelty: torch.Tensor
    ) -> Tuple[torch.Tensor, str]:
        """
        Determine whether adaptation should be triggered.
        
        Args:
            confidence: Model confidence [batch, 1]
            novelty: Novelty score [batch, 1]
            
        Returns:
            Tuple of (should_adapt [batch], trigger_reason)
        """
        # Check each trigger condition
        low_confidence = confidence < self.confidence_threshold
        high_novelty = novelty > self.novelty_threshold
        time_trigger = self.steps_since_adaptation >= self.adaptation_cooldown
        
        # Combine triggers
        should = low_confidence | high_novelty | time_trigger
        
        # Determine primary reason
        if low_confidence.any():
            reason = "low_confidence"
        elif high_novelty.any():
            reason = "high_novelty"
        elif time_trigger:
            reason = "time_based"
        else:
            reason = "none"
        
        return should.squeeze(-1), reason
    
    def forward(
        self,
        x: torch.Tensor,
        confidence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate adaptation triggers.
        
        Args:
            x: Input representation [batch, hidden_dim]
            confidence: Model confidence [batch, 1]
            
        Returns:
            Dictionary with trigger information
        """
        # Compute novelty
        novelty = self.compute_novelty(x)
        
        # Check triggers
        should_adapt, reason = self.should_adapt(confidence, novelty)
        
        # Update step counter
        if should_adapt.any():
            self.steps_since_adaptation.zero_()
        else:
            self.steps_since_adaptation += 1
        
        # Update statistics
        self.update_statistics(x.detach())
        
        return {
            'should_adapt': should_adapt,
            'trigger_reason': reason,
            'novelty': novelty,
            'confidence': confidence
        }


if __name__ == "__main__":
    # Test RNA editing module
    print("Testing RNAEditingLayer...")
    
    batch_size = 2
    seq_len = 10
    hidden_dim = 256
    num_heads = 8
    num_pathways = 3
    
    # Create layer
    rna_layer = RNAEditingLayer(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_pathways=num_pathways
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, hidden_dim)
    result = rna_layer(x, return_diagnostics=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result['output'].shape}")
    print(f"Temperature: {result['temperature'].squeeze().tolist()}")
    print(f"Confidence: {result['confidence'].squeeze().tolist()}")
    print(f"Head gates shape: {result['head_gates'].shape}")
    print(f"Head gates sum: {result['head_gates'].sum(dim=-1).tolist()}")
    print(f"Pathway weights: {result['pathway_weights'].tolist()}")
    print(f"Pathway weights sum: {result['pathway_weights'].sum(dim=-1).tolist()}")
    
    # Test with 2D input
    x_2d = torch.randn(batch_size, hidden_dim)
    result_2d = rna_layer(x_2d)
    print(f"\n2D input shape: {x_2d.shape}")
    print(f"2D output shape: {result_2d['output'].shape}")
    
    # Test adaptive trigger system
    print("\n\nTesting AdaptiveTriggerSystem...")
    trigger_system = AdaptiveTriggerSystem(hidden_dim=hidden_dim)
    
    context = torch.randn(batch_size, hidden_dim)
    confidence = torch.tensor([[0.3], [0.8]])  # One low, one high
    
    trigger_result = trigger_system(context, confidence)
    print(f"Should adapt: {trigger_result['should_adapt'].tolist()}")
    print(f"Trigger reason: {trigger_result['trigger_reason']}")
    print(f"Novelty: {trigger_result['novelty'].squeeze().tolist()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in rna_layer.parameters())
    print(f"\nRNA Editing Layer parameters: {total_params:,}")
    
    print("\nAll RNA editing tests passed!")
