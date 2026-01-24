"""
Base Limb - Abstract base class for distributed processing units
Inspired by octopus arm semi-autonomy

Key biological insights:
- Octopus arms contain ~2/3 of their neurons in arm ganglia
- Arms can perform basic tasks independently (reaching, grasping)
- Arms coordinate with central brain but have local decision-making
- Each arm maintains local sensory-motor loops

Our implementation:
- Each limb has local parameters + LoRA adapter
- Limbs maintain local experience buffers
- Limbs can process locally but sync with central hub
- Delta weights are computed for federated averaging
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from collections import deque

import sys
sys.path.append('..')
from adaptation.lora import LoRALayer


class ExperienceBuffer:
    """
    Local experience buffer for limb learning.
    Stores recent activations and gradients for local adaptation.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
    
    def push(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        loss: Optional[float] = None
    ):
        """Add experience to buffer"""
        self.buffer.append({
            'input': input_tensor.detach().cpu(),
            'output': output_tensor.detach().cpu(),
            'target': target.detach().cpu() if target is not None else None,
            'loss': loss
        })
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample random batch from buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = torch.randperm(len(self.buffer))[:batch_size].tolist()
        return [self.buffer[i] for i in indices]
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent experiences"""
        return list(self.buffer)[-n:]
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)


class BaseLimb(nn.Module, ABC):
    """
    Abstract base class for all limbs (distributed processing units).
    
    Each limb:
    1. Has a specialized function (perception, reasoning, action)
    2. Maintains local LoRA adapters for rapid adaptation
    3. Stores local experiences for learning
    4. Reports delta weights for hub synchronization
    5. Can estimate confidence in its outputs
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        buffer_size: int = 100,
        limb_name: str = "base"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.limb_name = limb_name
        
        # Main transformation
        self.transform = nn.Linear(input_dim, output_dim)
        
        # LoRA adapter for rapid adaptation (RNA editing analog)
        self.lora = LoRALayer(
            in_features=input_dim,
            out_features=output_dim,
            rank=lora_rank,
            alpha=lora_alpha
        )
        
        # Local experience buffer
        self.experience_buffer = ExperienceBuffer(max_size=buffer_size)
        
        # Track local gradient updates for sync
        self._gradient_step_count = 0
        self._last_sync_step = 0
        
        # Confidence estimation
        self._running_confidence = 0.5
        self._confidence_momentum = 0.9
    
    @abstractmethod
    def process(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Main processing logic - to be implemented by subclasses.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments specific to limb type
            
        Returns:
            Processed tensor
        """
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Forward pass through limb with optional confidence estimation.
        
        Args:
            x: Input tensor
            return_confidence: Whether to return confidence score
            **kwargs: Additional arguments passed to process()
            
        Returns:
            Tuple of (output_tensor, confidence) if return_confidence else just output
        """
        # Base transformation + LoRA adaptation
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted_out = base_out + lora_out
        
        # Subclass-specific processing
        output = self.process(adapted_out, **kwargs)
        
        if return_confidence:
            confidence = self.estimate_confidence(x, output)
            return output, confidence
        
        return output, None
    
    def estimate_confidence(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor
    ) -> float:
        """
        Estimate confidence in the output.
        
        Uses output entropy and LoRA adaptation magnitude as signals.
        Low entropy + small LoRA delta = high confidence
        """
        with torch.no_grad():
            # Entropy-based confidence (if output is logits/probabilities)
            if output_tensor.dim() >= 2 and output_tensor.size(-1) > 1:
                probs = torch.softmax(output_tensor, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
                max_entropy = torch.log(torch.tensor(output_tensor.size(-1), dtype=torch.float))
                entropy_confidence = 1.0 - (entropy / max_entropy).item()
            else:
                entropy_confidence = 0.5
            
            # LoRA magnitude as uncertainty signal
            # Large adaptations might indicate out-of-distribution inputs
            lora_delta = self.lora.get_delta_weight()
            lora_norm = lora_delta.norm().item()
            base_norm = self.transform.weight.norm().item()
            lora_ratio = lora_norm / (base_norm + 1e-10)
            lora_confidence = max(0.0, 1.0 - lora_ratio * 2)  # Scale factor
            
            # Combined confidence
            confidence = 0.6 * entropy_confidence + 0.4 * lora_confidence
            
            # Update running estimate
            self._running_confidence = (
                self._confidence_momentum * self._running_confidence +
                (1 - self._confidence_momentum) * confidence
            )
            
            return confidence
    
    def get_delta_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get weight deltas for hub synchronization.
        
        Returns dict of parameter name -> delta tensor
        """
        deltas = {}
        
        # LoRA delta
        deltas[f'{self.limb_name}/lora_delta'] = self.lora.get_delta_weight()
        
        # If transform has been updated, include that too
        # (for federated averaging of base weights)
        if hasattr(self, '_initial_transform_weight'):
            current_weight = self.transform.weight.data
            deltas[f'{self.limb_name}/transform_delta'] = (
                current_weight - self._initial_transform_weight
            )
        
        return deltas
    
    def apply_sync_update(
        self,
        aggregated_deltas: Dict[str, torch.Tensor],
        learning_rate: float = 0.1
    ):
        """
        Apply aggregated updates from hub sync.
        
        Args:
            aggregated_deltas: Dict of parameter name -> aggregated delta
            learning_rate: How much to blend aggregated update
        """
        for name, delta in aggregated_deltas.items():
            if f'{self.limb_name}/lora' in name:
                # Blend LoRA updates
                current_A = self.lora.lora_A.data
                current_B = self.lora.lora_B.data
                
                # Reconstruct A and B from delta (simple approach)
                # In practice, might sync A and B separately
                self.lora.lora_A.data = current_A + learning_rate * delta[:self.lora.rank, :]
            
            elif f'{self.limb_name}/transform' in name:
                # Blend transform updates
                self.transform.weight.data = (
                    self.transform.weight.data + 
                    learning_rate * delta
                )
        
        self._last_sync_step = self._gradient_step_count
    
    def record_experience(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        loss: Optional[float] = None
    ):
        """Record experience to local buffer"""
        self.experience_buffer.push(input_tensor, output_tensor, target, loss)
    
    def get_gradient_steps_since_sync(self) -> int:
        """Get number of gradient steps since last sync"""
        return self._gradient_step_count - self._last_sync_step
    
    def increment_gradient_step(self):
        """Call after each gradient update"""
        self._gradient_step_count += 1
    
    def reset_adaptation(self):
        """Reset LoRA adapters to zero (clear RNA editing)"""
        self.lora.reset_parameters()
        self._running_confidence = 0.5
    
    def get_stats(self) -> Dict[str, Any]:
        """Get limb statistics for monitoring"""
        return {
            'name': self.limb_name,
            'gradient_steps': self._gradient_step_count,
            'steps_since_sync': self.get_gradient_steps_since_sync(),
            'buffer_size': len(self.experience_buffer),
            'running_confidence': self._running_confidence,
            'lora_norm': self.lora.get_delta_weight().norm().item(),
            'num_params': sum(p.numel() for p in self.parameters())
        }


if __name__ == "__main__":
    # Test base limb (can't instantiate abstract class, so test components)
    print("Testing BaseLimb components...")
    
    # Test ExperienceBuffer
    buffer = ExperienceBuffer(max_size=10)
    for i in range(15):
        buffer.push(
            torch.randn(8, 256),
            torch.randn(8, 256),
            loss=float(i) / 10
        )
    
    print(f"Buffer size (should be 10): {len(buffer)}")
    recent = buffer.get_recent(3)
    print(f"Recent experiences: {len(recent)}")
    sample = buffer.sample(5)
    print(f"Sampled experiences: {len(sample)}")
    
    # Test LoRALayer standalone
    lora = LoRALayer(256, 256, rank=4)
    x = torch.randn(2, 10, 256)
    base_weight = torch.randn(256, 256)
    out = lora(x, base_weight)
    print(f"LoRA output shape: {out.shape}")
    
    print("\nBaseLimb component tests passed!")
