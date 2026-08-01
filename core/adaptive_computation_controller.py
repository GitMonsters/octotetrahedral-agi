"""
Adaptive Computation Time (ACT) Framework

Implements full adaptive computation control:
1. Budget allocation based on input complexity (RDT uncertainty)
2. Formal halting probabilities at each depth (ACT from Graves 2016)
3. Ponder cost tracking and regularization
4. Dynamic routing intensity based on remaining budget
5. Graceful degradation under computational pressure

Integration:
- Input: RDT uncertainty → Initial budget
- Loop: Halting probability + cost tracking
- Routing: Scale limb/braid intensity by budget remaining
- Loss: Add cost regularization term
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class AdaptiveComputationConfig:
    """Configuration for adaptive computation time."""
    
    # Budget allocation
    base_budget: float = 1.0              # Base ponder cost per loop (typically 0.5-1.5)
    budget_from_uncertainty: bool = True  # Scale budget by input uncertainty
    uncertainty_budget_scale: float = 2.0 # Max budget multiplier from uncertainty
    
    # Halting mechanism
    halting_hidden_dim: int = 128         # Hidden dim for halting gate
    halting_threshold: float = 0.99       # P(accumulated_halt) > threshold → stop
    
    # Cost regularization
    cost_loss_weight: float = 0.01        # Weight of ACT cost loss
    cost_target: float = 0.5              # Target average ponder cost (0-1)
    
    # Routing intensity
    intensity_scaling: bool = True        # Scale limb processing by budget ratio
    intensity_min: float = 0.3            # Min processing intensity when budgetexhausted
    intensity_max: float = 1.0            # Max processing intensity with budget available
    
    # Graceful degradation
    force_minimum_depth: int = 1          # Always run at least this many loops
    enable_budget_pressure: bool = True   # Enable budget pressure in routing


class HaltingGate(nn.Module):
    """
    Adaptive Computation Time halting gate (ACT from Graves 2016).
    
    At each step, learns probability of halting:
    - p_halt: Should we stop processing?
    - This is the "pondering" decision
    - Cumulative probability: P(halt by step t)
    - When P(halt) > threshold, stop reasoning
    """
    
    def __init__(self, hidden_dim: int, halting_hidden_dim: int = 128):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, halting_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(halting_hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        
        Returns:
            halting_prob: [batch] — probability of halting at this step
        """
        pooled = x.mean(dim=1)  # [batch, hidden_dim]
        halting_prob = self.gate(pooled).squeeze(-1)  # [batch]
        return halting_prob


class ComputationBudgetAllocator(nn.Module):
    """
    Allocates computation budget for each input based on complexity.
    
    Budget = base_budget * (1 + uncertainty * uncertainty_scale)
    
    Intuition:
    - High uncertainty → needs more compute
    - Low uncertainty → can compute quickly
    """
    
    def __init__(self, config: AdaptiveComputationConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        batch_size: int,
        uncertainties: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        """
        Allocate budget for batch.
        
        Args:
            batch_size: Number of samples
            uncertainties: [batch] — RDT uncertainty (0-1), or None for uniform
            device: Device to create tensor on
        
        Returns:
            budgets: [batch] — computation budget for each sample
        """
        budgets = torch.full((batch_size,), self.config.base_budget, device=device)
        
        if self.config.budget_from_uncertainty and uncertainties is not None:
            # Scale budget by uncertainty: high uncertainty → high budget
            # budget = base * (1 + uncertainty * scale)
            uncertainty_factor = 1.0 + uncertainties * (self.config.uncertainty_budget_scale - 1.0)
            budgets = budgets * uncertainty_factor
        
        return budgets


class PonderCostTracker:
    """
    Tracks cumulative ponder cost across loop iterations.
    
    Ponder cost = time spent reasoning (each loop step costs base_budget)
    Goal: Keep average ponder cost close to target (e.g., 0.5)
    
    Too low (early exit): Model might not think enough
    Too high (late exit): Model wastes compute on easy problems
    """
    
    def __init__(self, config: AdaptiveComputationConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset tracking for new batch."""
        self.cumulative_halt = None  # [batch] — P(halt by step t)
        self.ponder_costs = []        # List of costs per step
        self.halting_probs = []       # List of halting probs per step
    
    def step(
        self,
        halting_prob: torch.Tensor,
        step_cost: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Record one step of computation.
        
        Args:
            halting_prob: [batch] — p(halt at this step)
            step_cost: Cost of this step (default 1.0 per loop)
        
        Returns:
            cumulative_halt: [batch] — P(halt by this step)
            alive_prob: [batch] — P(still computing | hasn't halted)
        """
        batch_size = halting_prob.shape[0]
        device = halting_prob.device
        
        if self.cumulative_halt is None:
            self.cumulative_halt = torch.zeros(batch_size, device=device)
            self.alive_prob = torch.ones(batch_size, device=device)
        
        # P(halt at this step) = P(still alive) * P(halt | alive)
        halt_at_step = self.alive_prob * halting_prob
        
        # Update cumulative halt probability
        self.cumulative_halt = self.cumulative_halt + halt_at_step
        
        # Update alive probability: P(alive after this step) = P(alive) * (1 - halt)
        self.alive_prob = self.alive_prob * (1.0 - halting_prob)
        
        # Track for loss computation
        self.halting_probs.append(halting_prob)
        self.ponder_costs.append(step_cost * halt_at_step)
        
        return self.cumulative_halt.clone(), self.alive_prob.clone()
    
    def get_ponder_cost(self) -> torch.Tensor:
        """
        Compute total ponder cost across all steps.
        
        Returns:
            ponder_cost: [batch] — total compute used per sample
        """
        if not self.ponder_costs:
            return torch.tensor(0.0)
        
        # Stack costs: [num_steps, batch]
        costs = torch.stack(self.ponder_costs, dim=0)
        
        # Sum across steps: [batch]
        total_cost = costs.sum(dim=0)
        
        return total_cost
    
    def get_cost_loss(self, target_cost: Optional[float] = None) -> torch.Tensor:
        """
        Compute ACT cost regularization loss.
        
        Encourages average ponder cost to match target:
        loss = |mean(ponder_cost) - target|^2
        
        Args:
            target_cost: Target average cost (default from config)
        
        Returns:
            loss: Scalar loss
        """
        if not self.ponder_costs:
            return torch.tensor(0.0)
        
        target = target_cost or self.config.cost_target
        total_cost = self.get_ponder_cost()
        
        # MSE between average cost and target
        avg_cost = total_cost.mean()
        cost_loss = (avg_cost - target) ** 2
        
        return cost_loss


class AdaptiveComputationController(nn.Module):
    """
    Full adaptive computation time controller.
    
    Manages:
    1. Budget allocation (RDT uncertainty → budget)
    2. Halting decisions (formal ACT halting gates)
    3. Cost tracking (ponder cost regularization)
    4. Routing intensity (scale by remaining budget)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[AdaptiveComputationConfig] = None,
        max_loops: int = 4,
    ):
        super().__init__()
        
        self.config = config or AdaptiveComputationConfig()
        self.hidden_dim = hidden_dim
        self.max_loops = max_loops
        
        # Budget allocator
        self.budget_allocator = ComputationBudgetAllocator(self.config)
        
        # Halting gates for each loop (last loop always halts)
        self.halting_gates = nn.ModuleList([
            HaltingGate(hidden_dim, self.config.halting_hidden_dim)
            for _ in range(max(1, max_loops - 1))
        ])
        
        # Cost tracker (not a nn.Module, stateful)
        self._cost_tracker = None
        
        # Statistics
        self._last_budgets = None
        self._last_ponder_costs = None
        self._last_routing_intensities = None
    
    def allocate_budget(
        self,
        batch_size: int,
        rdt_uncertainties: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        """
        Allocate computation budget for batch.
        
        Args:
            batch_size: Number of samples
            rdt_uncertainties: [batch] — RDT uncertainty from previous pass
            device: Device to create tensor on
        
        Returns:
            budgets: [batch] — computation budget for each sample
        """
        budgets = self.budget_allocator(batch_size, rdt_uncertainties, device)
        self._last_budgets = budgets
        return budgets
    
    def start_computation(self, batch_size: int):
        """Initialize cost tracker for new forward pass."""
        self._cost_tracker = PonderCostTracker(self.config)
    
    def should_halt(
        self,
        state: torch.Tensor,
        loop_idx: int,
        budgets: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Determine halting probability at this loop step.
        
        Args:
            state: [batch, seq_len, hidden_dim] — current processing state
            loop_idx: Current loop iteration (0-indexed)
            budgets: [batch] — remaining budget for each sample
            training: Whether in training mode
        
        Returns:
            halting_prob: [batch] — probability of halting
            cumulative_halt: [batch] — P(halt by this step)
            alive_prob: [batch] — P(still computing)
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Get halting probability from gate
        is_last = (loop_idx == self.max_loops - 1)
        if is_last:
            # Force halt on last step
            halting_prob = torch.ones(batch_size, device=device)
        else:
            halting_prob = self.halting_gates[loop_idx](state)  # [batch]
            
            # Apply budget pressure if enabled
            if self.config.enable_budget_pressure and budgets is not None:
                # Remaining budget as fraction of original
                budget_ratio = budgets / (self.config.base_budget * self.config.uncertainty_budget_scale)
                budget_ratio = torch.clamp(budget_ratio, 0.0, 1.0)
                
                # High budget → lower halting prob (keep computing)
                # Low budget → higher halting prob (force halt)
                budget_pressure = 1.0 - budget_ratio  # [batch]
                
                # Blend: high pressure increases halting prob
                halting_prob = 0.7 * halting_prob + 0.3 * budget_pressure
                halting_prob = torch.clamp(halting_prob, 0.0, 1.0 - 1e-6)
        
        # Track cost
        if self._cost_tracker is not None:
            cumulative_halt, alive_prob = self._cost_tracker.step(halting_prob, step_cost=1.0)
        else:
            cumulative_halt = halting_prob
            alive_prob = 1.0 - halting_prob
        
        return halting_prob, cumulative_halt, alive_prob
    
    def get_routing_intensity(
        self,
        loop_idx: int,
        budgets: Optional[torch.Tensor] = None,
        max_loops: int = 4,
    ) -> torch.Tensor:
        """
        Compute routing intensity based on remaining budget.
        
        Intensity scales limb/braid processing:
        - High budget → intensity = 1.0 (full processing)
        - Low budget → intensity = intensity_min (sparse processing)
        
        Args:
            loop_idx: Current loop iteration
            budgets: [batch] — remaining budget for each sample
            max_loops: Maximum loops available
        
        Returns:
            intensity: [batch] — routing intensity (0-1)
        """
        batch_size = budgets.shape[0] if budgets is not None else 1
        device = budgets.device if budgets is not None else torch.device('cpu')
        
        if budgets is None or not self.config.intensity_scaling:
            # No scaling: full intensity
            return torch.ones(batch_size, device=device)
        
        # Cost so far
        cost_so_far = self._cost_tracker.get_ponder_cost() if self._cost_tracker else torch.zeros(batch_size, device=device)
        
        # Remaining budget as fraction
        original_budget = self.config.base_budget * self.config.uncertainty_budget_scale
        remaining_fraction = (budgets - cost_so_far) / original_budget
        remaining_fraction = torch.clamp(remaining_fraction, 0.0, 1.0)
        
        # Map to intensity range: [intensity_min, intensity_max]
        intensity = (
            self.config.intensity_min +
            remaining_fraction * (self.config.intensity_max - self.config.intensity_min)
        )
        
        self._last_routing_intensities = intensity
        return intensity
    
    def get_cost_loss(self) -> torch.Tensor:
        """
        Get ACT cost regularization loss.
        
        Returns:
            loss: Scalar tensor
        """
        if self._cost_tracker is None:
            return torch.tensor(0.0)
        
        cost_loss = self._cost_tracker.get_cost_loss(self.config.cost_target)
        return self.config.cost_loss_weight * cost_loss
    
    def get_ponder_cost(self) -> torch.Tensor:
        """
        Get total ponder cost for current batch.
        
        Returns:
            ponder_cost: [batch]
        """
        if self._cost_tracker is None:
            return torch.tensor(0.0)
        
        cost = self._cost_tracker.get_ponder_cost()
        self._last_ponder_costs = cost
        return cost
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics for monitoring."""
        stats = {
            'budgets': self._last_budgets.tolist() if self._last_budgets is not None else None,
            'ponder_costs': self._last_ponder_costs.tolist() if self._last_ponder_costs is not None else None,
            'routing_intensities': self._last_routing_intensities.mean().item() if self._last_routing_intensities is not None else None,
        }
        
        if self._cost_tracker is not None:
            stats['num_steps'] = len(self._cost_tracker.halting_probs)
            stats['halt_distribution'] = [p.mean().item() for p in self._cost_tracker.halting_probs]
        
        return stats
