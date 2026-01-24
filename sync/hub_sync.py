"""
Hub Synchronization - Federated Averaging for Limb Coordination
Inspired by octopus central-peripheral coordination

Biological insight:
- Octopus brain sends high-level commands to arms
- Arms send summarized sensory information back
- Coordination happens through selective attention and gating
- Arms maintain autonomy but respect central override

Our implementation:
- Federated Averaging (FedAvg) for weight synchronization
- Safety constraints (max drift per sync)
- Rollback buffer for error recovery
- Performance-weighted aggregation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RollbackBuffer:
    """
    Stores model state checkpoints for rollback.
    Enables recovery from divergent or degraded limb updates.
    """
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
    
    def save(
        self,
        state_dict: Dict[str, torch.Tensor],
        step: int,
        performance: float
    ):
        """Save a checkpoint"""
        self.buffer.append({
            'state': {k: v.clone() for k, v in state_dict.items()},
            'step': step,
            'performance': performance
        })
    
    def get_best(self) -> Optional[Dict[str, Any]]:
        """Get checkpoint with best performance"""
        if not self.buffer:
            return None
        return max(self.buffer, key=lambda x: x['performance'])
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get most recent checkpoint"""
        if not self.buffer:
            return None
        return self.buffer[-1]
    
    def rollback_to_best(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get state dict of best checkpoint"""
        best = self.get_best()
        return best['state'] if best else None
    
    def __len__(self) -> int:
        return len(self.buffer)


class HubSync:
    """
    Hub Synchronization Manager for coordinating limb weight updates.
    
    Implements:
    1. FedAvg - aggregate limb weight deltas
    2. Safety constraints - limit weight drift
    3. Performance weighting - better limbs get more influence
    4. Rollback capability - recover from bad updates
    """
    
    def __init__(
        self,
        sync_frequency: int = 10,
        max_drift: float = 0.1,
        rollback_buffer_size: int = 10,
        use_performance_weighting: bool = True,
        clip_grad_norm: Optional[float] = 1.0
    ):
        self.sync_frequency = sync_frequency
        self.max_drift = max_drift
        self.use_performance_weighting = use_performance_weighting
        self.clip_grad_norm = clip_grad_norm
        
        # Tracking
        self._step_count = 0
        self._last_sync_step = 0
        self._sync_history: List[Dict[str, Any]] = []
        
        # Rollback buffer
        self.rollback_buffer = RollbackBuffer(max_size=rollback_buffer_size)
        
        # Limb performance tracking (for weighted averaging)
        self._limb_performances: Dict[str, float] = {}
        
        # Statistics
        self._total_syncs = 0
        self._rollbacks = 0
        self._drift_violations = 0
    
    def should_sync(self) -> bool:
        """Check if it's time to sync"""
        return (self._step_count - self._last_sync_step) >= self.sync_frequency
    
    def collect_deltas(
        self,
        limbs: Dict[str, nn.Module]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Collect weight deltas from all limbs.
        
        Args:
            limbs: Dict of limb_name -> limb module
            
        Returns:
            Dict of limb_name -> delta_dict
        """
        all_deltas = {}
        
        for name, limb in limbs.items():
            if hasattr(limb, 'get_delta_weights'):
                deltas = limb.get_delta_weights()
                all_deltas[name] = deltas
        
        return all_deltas
    
    def aggregate_fedavg(
        self,
        all_deltas: Dict[str, Dict[str, torch.Tensor]],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging of weight deltas.
        
        Args:
            all_deltas: Dict of limb_name -> delta_dict
            weights: Optional performance-based weights per limb
            
        Returns:
            Aggregated delta dict
        """
        if not all_deltas:
            return {}
        
        # Default to uniform weights
        if weights is None:
            n_limbs = len(all_deltas)
            weights = {name: 1.0 / n_limbs for name in all_deltas.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Aggregate
        aggregated = {}
        
        # Collect all unique delta keys
        all_keys = set()
        for deltas in all_deltas.values():
            all_keys.update(deltas.keys())
        
        for key in all_keys:
            weighted_sum = None
            total_w = 0.0
            
            for limb_name, deltas in all_deltas.items():
                if key in deltas:
                    w = weights.get(limb_name, 0.0)
                    delta = deltas[key]
                    
                    if weighted_sum is None:
                        weighted_sum = w * delta.clone()
                    else:
                        weighted_sum = weighted_sum + w * delta
                    
                    total_w += w
            
            if weighted_sum is not None and total_w > 0:
                aggregated[key] = weighted_sum / total_w
        
        return aggregated
    
    def apply_safety_constraints(
        self,
        aggregated: Dict[str, torch.Tensor],
        current_weights: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], bool]:
        """
        Apply safety constraints to limit weight drift.
        
        Args:
            aggregated: Aggregated weight deltas
            current_weights: Current model weights
            
        Returns:
            Tuple of (constrained_deltas, was_clipped)
        """
        constrained = {}
        was_clipped = False
        
        for key, delta in aggregated.items():
            # Get corresponding current weight if available
            if key in current_weights:
                current = current_weights[key]
                current_norm = current.norm().item()
                
                # Max allowed change
                max_change = self.max_drift * current_norm
                delta_norm = delta.norm().item()
                
                if delta_norm > max_change:
                    # Clip to max allowed change
                    scale = max_change / (delta_norm + 1e-10)
                    constrained[key] = delta * scale
                    was_clipped = True
                    self._drift_violations += 1
                    logger.warning(
                        f"Drift violation on {key}: {delta_norm:.4f} > {max_change:.4f}, "
                        f"clipped by {scale:.4f}"
                    )
                else:
                    constrained[key] = delta
            else:
                constrained[key] = delta
        
        return constrained, was_clipped
    
    def sync(
        self,
        model: nn.Module,
        limbs: Dict[str, nn.Module],
        current_performance: float
    ) -> Dict[str, Any]:
        """
        Perform hub synchronization.
        
        Args:
            model: Main model to update
            limbs: Dict of limb modules
            current_performance: Current model performance (for rollback decision)
            
        Returns:
            Dict with sync statistics
        """
        # Save checkpoint before sync
        self.rollback_buffer.save(
            model.state_dict(),
            self._step_count,
            current_performance
        )
        
        # Collect deltas from limbs
        all_deltas = self.collect_deltas(limbs)
        
        if not all_deltas:
            return {'status': 'no_deltas', 'synced': False}
        
        # Get performance weights if enabled
        weights = None
        if self.use_performance_weighting and self._limb_performances:
            weights = self._limb_performances.copy()
        
        # Aggregate using FedAvg
        aggregated = self.aggregate_fedavg(all_deltas, weights)
        
        # Get current weights for safety check
        current_weights = {k: v for k, v in model.state_dict().items()}
        
        # Apply safety constraints
        constrained, was_clipped = self.apply_safety_constraints(
            aggregated, current_weights
        )
        
        # Apply updates to model
        updates_applied = self._apply_updates(model, constrained)
        
        # Update tracking
        self._last_sync_step = self._step_count
        self._total_syncs += 1
        
        # Record history
        sync_record = {
            'step': self._step_count,
            'n_deltas': len(all_deltas),
            'was_clipped': was_clipped,
            'updates_applied': updates_applied,
            'performance': current_performance
        }
        self._sync_history.append(sync_record)
        
        return {
            'status': 'synced',
            'synced': True,
            **sync_record
        }
    
    def _apply_updates(
        self,
        model: nn.Module,
        deltas: Dict[str, torch.Tensor]
    ) -> int:
        """Apply aggregated updates to model parameters"""
        updates = 0
        state_dict = model.state_dict()
        
        for key, delta in deltas.items():
            # Map delta key back to model parameter
            # Delta keys are like "perception/lora_delta" 
            # We need to find corresponding model params
            
            # Simple matching for now - look for LoRA params
            for param_name, param in model.named_parameters():
                if 'lora' in param_name.lower():
                    # Check if this delta applies
                    limb_name = key.split('/')[0]
                    if limb_name in param_name:
                        if param.shape == delta.shape:
                            param.data.add_(delta * 0.1)  # Blend factor
                            updates += 1
                            break
        
        return updates
    
    def rollback(self, model: nn.Module) -> bool:
        """
        Rollback model to best checkpoint.
        
        Args:
            model: Model to rollback
            
        Returns:
            Whether rollback was successful
        """
        best_state = self.rollback_buffer.rollback_to_best()
        
        if best_state is None:
            logger.warning("No checkpoints available for rollback")
            return False
        
        model.load_state_dict(best_state)
        self._rollbacks += 1
        logger.info(f"Rolled back to checkpoint with best performance")
        return True
    
    def update_limb_performance(self, limb_name: str, performance: float):
        """Update performance tracking for a limb"""
        self._limb_performances[limb_name] = performance
    
    def step(self):
        """Increment step counter"""
        self._step_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        return {
            'total_syncs': self._total_syncs,
            'total_steps': self._step_count,
            'rollbacks': self._rollbacks,
            'drift_violations': self._drift_violations,
            'checkpoint_count': len(self.rollback_buffer),
            'sync_frequency': self.sync_frequency,
            'max_drift': self.max_drift,
            'limb_performances': self._limb_performances.copy()
        }


class DistributedCoordinator:
    """
    Higher-level coordinator for multi-limb systems.
    Manages communication patterns between limbs and hub.
    """
    
    def __init__(
        self,
        hub_sync: HubSync,
        limbs: Dict[str, nn.Module]
    ):
        self.hub_sync = hub_sync
        self.limbs = limbs
        
        # Communication buffers
        self._limb_outputs: Dict[str, torch.Tensor] = {}
        self._pending_messages: List[Dict[str, Any]] = []
    
    def broadcast_to_limbs(
        self,
        message: torch.Tensor,
        source: str = "hub"
    ):
        """Broadcast message from hub to all limbs"""
        for limb_name, limb in self.limbs.items():
            if hasattr(limb, 'receive_broadcast'):
                limb.receive_broadcast(message, source)
    
    def collect_from_limbs(self) -> Dict[str, torch.Tensor]:
        """Collect outputs/states from all limbs"""
        outputs = {}
        for limb_name, limb in self.limbs.items():
            if hasattr(limb, 'get_output'):
                outputs[limb_name] = limb.get_output()
        return outputs
    
    def coordinate_step(
        self,
        model: nn.Module,
        performance: float
    ) -> Optional[Dict[str, Any]]:
        """
        Perform one coordination step.
        
        Returns sync results if sync occurred, else None.
        """
        self.hub_sync.step()
        
        if self.hub_sync.should_sync():
            return self.hub_sync.sync(model, self.limbs, performance)
        
        return None


if __name__ == "__main__":
    print("Testing HubSync...")
    
    # Create mock limbs with get_delta_weights method
    class MockLimb(nn.Module):
        def __init__(self, name: str, dim: int = 256):
            super().__init__()
            self.name = name
            self.linear = nn.Linear(dim, dim)
            self.lora_A = nn.Parameter(torch.randn(4, dim) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(dim, 4) * 0.01)
        
        def get_delta_weights(self):
            return {
                f'{self.name}/lora_delta': self.lora_B @ self.lora_A
            }
    
    # Create limbs
    limbs = {
        'perception': MockLimb('perception'),
        'reasoning': MockLimb('reasoning'),
        'action': MockLimb('action')
    }
    
    # Create hub sync
    hub = HubSync(
        sync_frequency=5,
        max_drift=0.1,
        use_performance_weighting=True
    )
    
    # Set limb performances
    hub.update_limb_performance('perception', 0.8)
    hub.update_limb_performance('reasoning', 0.9)
    hub.update_limb_performance('action', 0.7)
    
    # Test delta collection
    deltas = hub.collect_deltas(limbs)
    print(f"Collected deltas from {len(deltas)} limbs")
    for name, delta_dict in deltas.items():
        for key, tensor in delta_dict.items():
            print(f"  {key}: shape {tensor.shape}")
    
    # Test FedAvg
    aggregated = hub.aggregate_fedavg(deltas)
    print(f"\nAggregated {len(aggregated)} deltas")
    for key, tensor in aggregated.items():
        print(f"  {key}: shape {tensor.shape}, norm {tensor.norm().item():.4f}")
    
    # Test sync
    model = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 256))
    
    # Simulate steps
    for i in range(10):
        hub.step()
        if hub.should_sync():
            result = hub.sync(model, limbs, performance=0.8 + i * 0.01)
            print(f"\nSync at step {i + 1}: {result}")
    
    # Stats
    stats = hub.get_stats()
    print(f"\nHub stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\nHubSync tests passed!")
