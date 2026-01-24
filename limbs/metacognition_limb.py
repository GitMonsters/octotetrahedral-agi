"""
MetaCognition Limb - Self-monitoring, introspection, and meta-learning
Inspired by octopus self-awareness and adaptive behavior

Biological insight:
- Octopuses show signs of self-awareness (mirror tests are complex)
- Demonstrate adaptive learning strategies
- Can modify behavior based on outcome monitoring
- Show individual personalities and learning styles

Our implementation:
- Confidence calibration and uncertainty estimation
- Learning rate and strategy adaptation
- Self-model maintenance
- Meta-learning for hyperparameter adjustment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from collections import deque

from .base_limb import BaseLimb


class UncertaintyEstimator(nn.Module):
    """
    Estimates epistemic and aleatoric uncertainty.
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Epistemic uncertainty (model uncertainty)
        self.epistemic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive
        )
        
        # Aleatoric uncertainty (data uncertainty)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Combined uncertainty
        self.combiner = nn.Linear(2, 1)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainties from features.
        
        Args:
            features: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            
        Returns:
            Dict with epistemic, aleatoric, and total uncertainty
        """
        # Pool if sequence
        if features.dim() == 3:
            features = features.mean(dim=1)
        
        epistemic = self.epistemic_head(features)
        aleatoric = self.aleatoric_head(features)
        
        combined = torch.cat([epistemic, aleatoric], dim=-1)
        total = self.combiner(combined)
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total
        }


class ConfidenceCalibrator(nn.Module):
    """
    Calibrates confidence estimates to match actual accuracy.
    """
    
    def __init__(self, hidden_dim: int = 256, num_bins: int = 10):
        super().__init__()
        
        self.num_bins = num_bins
        
        # Calibration network
        self.calibrator = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for raw confidence
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Track calibration statistics
        self.register_buffer('bin_counts', torch.zeros(num_bins))
        self.register_buffer('bin_correct', torch.zeros(num_bins))
        self.register_buffer('bin_confidence', torch.zeros(num_bins))
    
    def forward(
        self,
        features: torch.Tensor,
        raw_confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Calibrate raw confidence estimate.
        
        Args:
            features: [batch, hidden_dim]
            raw_confidence: [batch, 1] raw confidence
            
        Returns:
            Calibrated confidence [batch, 1]
        """
        combined = torch.cat([features, raw_confidence], dim=-1)
        calibrated = self.calibrator(combined)
        return calibrated
    
    def update_statistics(
        self,
        confidence: torch.Tensor,
        correct: torch.Tensor
    ):
        """Update calibration statistics."""
        with torch.no_grad():
            # Bin confidences
            bins = (confidence * self.num_bins).long().clamp(0, self.num_bins - 1)
            
            for b in range(self.num_bins):
                mask = bins == b
                self.bin_counts[b] += mask.sum()
                self.bin_correct[b] += (mask & correct).sum()
                self.bin_confidence[b] += confidence[mask].sum()
    
    def expected_calibration_error(self) -> float:
        """Compute Expected Calibration Error."""
        with torch.no_grad():
            total = self.bin_counts.sum()
            if total == 0:
                return 0.0
            
            accuracies = self.bin_correct / (self.bin_counts + 1e-10)
            avg_confidences = self.bin_confidence / (self.bin_counts + 1e-10)
            
            ece = (self.bin_counts / total * (accuracies - avg_confidences).abs()).sum()
            return ece.item()


class LearningStrategyAdapter(nn.Module):
    """
    Adapts learning strategy based on task characteristics and performance.
    """
    
    def __init__(self, hidden_dim: int = 256, num_strategies: int = 4):
        super().__init__()
        
        self.num_strategies = num_strategies
        self.strategy_names = [
            'exploration',  # High learning rate, more randomness
            'exploitation', # Low learning rate, greedy
            'consolidation', # Very low LR, reinforce learned patterns
            'adaptation'    # Medium LR, balanced
        ]
        
        # Strategy selector
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_strategies)
        )
        
        # Learning rate predictor
        self.lr_predictor = nn.Sequential(
            nn.Linear(hidden_dim + num_strategies, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1], scale later
        )
        
        # Temperature predictor (for sampling)
        self.temp_predictor = nn.Sequential(
            nn.Linear(hidden_dim + num_strategies, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
    
    def forward(
        self,
        features: torch.Tensor,
        performance_history: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Recommend learning strategy and hyperparameters.
        
        Args:
            features: Current state features [batch, hidden_dim]
            performance_history: Optional recent performance [batch, history_len]
            
        Returns:
            Dict with strategy, learning_rate, temperature
        """
        # Select strategy
        strategy_logits = self.selector(features)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        strategy_idx = strategy_logits.argmax(dim=-1)
        
        # Get strategy one-hot
        strategy_onehot = F.one_hot(strategy_idx, self.num_strategies).float()
        
        # Predict hyperparameters
        combined = torch.cat([features, strategy_onehot], dim=-1)
        
        lr_raw = self.lr_predictor(combined)
        lr = 1e-5 + lr_raw * (1e-2 - 1e-5)  # Scale to [1e-5, 1e-2]
        
        temp = self.temp_predictor(combined)
        temp = 0.1 + temp  # Ensure minimum temperature
        
        # Get strategy names
        strategies = [self.strategy_names[i] for i in strategy_idx.tolist()]
        
        return {
            'strategy': strategies,
            'strategy_probs': strategy_probs,
            'learning_rate': lr,
            'temperature': temp
        }


class SelfModel(nn.Module):
    """
    Maintains a model of the agent's own capabilities and state.
    """
    
    def __init__(self, hidden_dim: int = 256, num_capabilities: int = 8):
        super().__init__()
        
        self.num_capabilities = num_capabilities
        self.capability_names = [
            'perception', 'reasoning', 'memory', 'planning',
            'language', 'spatial', 'action', 'metacognition'
        ]
        
        # Capability estimator
        self.capability_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_capabilities),
            nn.Sigmoid()
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Track performance history per capability
        self.performance_history: Dict[str, deque] = {
            name: deque(maxlen=100) for name in self.capability_names
        }
    
    def forward(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        Estimate own capabilities and state.
        
        Args:
            features: [batch, hidden_dim]
            
        Returns:
            Dict with capabilities, state_encoding, strengths, weaknesses
        """
        # Estimate capabilities
        capabilities = self.capability_estimator(features)
        
        # Encode current state
        state_encoding = self.state_encoder(features)
        
        # Identify strengths and weaknesses
        cap_values = capabilities.mean(dim=0)  # Average across batch
        sorted_indices = cap_values.argsort(descending=True)
        
        strengths = [self.capability_names[i] for i in sorted_indices[:3].tolist()]
        weaknesses = [self.capability_names[i] for i in sorted_indices[-3:].tolist()]
        
        return {
            'capabilities': capabilities,
            'capability_names': self.capability_names,
            'state_encoding': state_encoding,
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    def update_performance(self, capability: str, performance: float):
        """Update performance history for a capability."""
        if capability in self.performance_history:
            self.performance_history[capability].append(performance)
    
    def get_capability_trend(self, capability: str) -> float:
        """Get performance trend for a capability."""
        if capability not in self.performance_history:
            return 0.0
        
        history = list(self.performance_history[capability])
        if len(history) < 2:
            return 0.0
        
        # Simple linear trend
        recent = sum(history[-10:]) / max(len(history[-10:]), 1)
        older = sum(history[:-10]) / max(len(history[:-10]), 1) if len(history) > 10 else recent
        
        return recent - older


class MetaCognitionLimb(BaseLimb):
    """
    MetaCognition Limb for self-monitoring and meta-learning.
    
    Capabilities:
    1. Uncertainty estimation (epistemic + aleatoric)
    2. Confidence calibration
    3. Learning strategy adaptation
    4. Self-model maintenance
    5. Introspection and explanation
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        buffer_size: int = 100
    ):
        super().__init__(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            buffer_size=buffer_size,
            limb_name="metacognition"
        )
        
        self.hidden_dim = hidden_dim
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(hidden_dim)
        
        # Confidence calibrator
        self.confidence_calibrator = ConfidenceCalibrator(hidden_dim)
        
        # Learning strategy adapter
        self.strategy_adapter = LearningStrategyAdapter(hidden_dim)
        
        # Self-model
        self.self_model = SelfModel(hidden_dim)
        
        # Introspection attention
        self.introspection_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Meta-feature extractor
        self.meta_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Reasoning about reasoning
        self.meta_reasoning = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Stats
        self._introspections = 0
        self._strategy_changes = 0
        self._current_strategy = 'exploration'
    
    def process(
        self,
        x: torch.Tensor,
        other_limb_outputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Process with meta-cognition.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
            other_limb_outputs: Optional outputs from other limbs
            
        Returns:
            Meta-enhanced features
        """
        batch_size = x.size(0)
        
        # Extract meta-features
        meta_features = self.meta_encoder(x)
        
        # Self-attention for introspection
        introspected, _ = self.introspection_attn(
            meta_features, meta_features, meta_features
        )
        
        # Combine with original
        combined = torch.cat([x, introspected], dim=-1)
        reasoned = self.meta_reasoning(combined)
        
        # Output
        output = self.norm(x + reasoned)
        output = self.output_proj(output)
        
        self._introspections += batch_size
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        other_limb_outputs: Optional[Dict[str, torch.Tensor]] = None,
        return_confidence: bool = False,
        return_uncertainty: bool = False,
        return_strategy: bool = False,
        return_self_model: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[Dict]]:
        """
        Forward pass through metacognition limb.
        """
        # Base transformation + LoRA
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        
        # Meta-cognitive processing
        output = self.process(adapted, other_limb_outputs=other_limb_outputs, **kwargs)
        
        # Collect extras
        extras = {}
        
        # Pool for single-vector analyses
        pooled = output.mean(dim=1) if output.dim() == 3 else output
        
        # Confidence
        confidence = None
        if return_confidence:
            # Get raw confidence
            raw_conf = self.estimate_confidence(x, output)
            raw_conf_tensor = torch.tensor([[raw_conf]], device=pooled.device).expand(pooled.size(0), 1)
            
            # Calibrate
            calibrated = self.confidence_calibrator(pooled, raw_conf_tensor)
            confidence = calibrated.mean().item()
            extras['raw_confidence'] = raw_conf
            extras['calibrated_confidence'] = confidence
        
        # Uncertainty
        if return_uncertainty:
            uncertainties = self.uncertainty_estimator(pooled)
            extras['uncertainty'] = {
                'epistemic': uncertainties['epistemic'].mean().item(),
                'aleatoric': uncertainties['aleatoric'].mean().item(),
                'total': uncertainties['total'].mean().item()
            }
        
        # Strategy
        if return_strategy:
            strategy_info = self.strategy_adapter(pooled)
            extras['strategy'] = strategy_info['strategy']
            extras['learning_rate'] = strategy_info['learning_rate'].mean().item()
            extras['temperature'] = strategy_info['temperature'].mean().item()
            
            # Track strategy changes
            if strategy_info['strategy'][0] != self._current_strategy:
                self._strategy_changes += 1
                self._current_strategy = strategy_info['strategy'][0]
        
        # Self-model
        if return_self_model:
            self_info = self.self_model(pooled)
            extras['capabilities'] = {
                name: self_info['capabilities'][:, i].mean().item()
                for i, name in enumerate(self_info['capability_names'])
            }
            extras['strengths'] = self_info['strengths']
            extras['weaknesses'] = self_info['weaknesses']
        
        return output, confidence, extras if extras else None
    
    def introspect(
        self,
        features: torch.Tensor,
        question: str = "general"
    ) -> Dict[str, Any]:
        """
        Perform deep introspection on current state.
        
        Args:
            features: Current features [batch, hidden_dim] or [batch, seq, hidden]
            question: Type of introspection
            
        Returns:
            Introspection results
        """
        if features.dim() == 3:
            features = features.mean(dim=1)
        
        results = {}
        
        # Always get uncertainty
        uncertainties = self.uncertainty_estimator(features)
        results['uncertainty'] = {
            'epistemic': uncertainties['epistemic'].mean().item(),
            'aleatoric': uncertainties['aleatoric'].mean().item(),
            'total': uncertainties['total'].mean().item()
        }
        
        # Get self-assessment
        self_info = self.self_model(features)
        results['capabilities'] = {
            name: self_info['capabilities'][:, i].mean().item()
            for i, name in enumerate(self_info['capability_names'])
        }
        results['strengths'] = self_info['strengths']
        results['weaknesses'] = self_info['weaknesses']
        
        # Get recommended strategy
        strategy_info = self.strategy_adapter(features)
        results['recommended_strategy'] = strategy_info['strategy'][0]
        results['recommended_lr'] = strategy_info['learning_rate'].mean().item()
        
        # Calibration error
        results['calibration_error'] = self.confidence_calibrator.expected_calibration_error()
        
        return results
    
    def should_explore(self, features: torch.Tensor) -> bool:
        """Determine if the model should explore vs exploit."""
        if features.dim() == 3:
            features = features.mean(dim=1)
        
        uncertainties = self.uncertainty_estimator(features)
        epistemic = uncertainties['epistemic'].mean().item()
        
        # High epistemic uncertainty -> should explore
        return epistemic > 0.5
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metacognition limb statistics."""
        stats = super().get_stats()
        stats.update({
            'introspections': self._introspections,
            'strategy_changes': self._strategy_changes,
            'current_strategy': self._current_strategy,
            'calibration_error': self.confidence_calibrator.expected_calibration_error()
        })
        return stats


if __name__ == "__main__":
    print("Testing MetaCognitionLimb...")
    
    # Create limb
    limb = MetaCognitionLimb(
        hidden_dim=256,
        num_heads=4
    )
    
    # Test input
    batch_size = 2
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 256)
    
    # Forward pass with all extras
    output, confidence, extras = limb(
        x,
        return_confidence=True,
        return_uncertainty=True,
        return_strategy=True,
        return_self_model=True
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Confidence: {confidence:.4f}")
    print(f"\nUncertainty:")
    for k, v in extras['uncertainty'].items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nRecommended strategy: {extras['strategy']}")
    print(f"Recommended LR: {extras['learning_rate']:.6f}")
    print(f"Recommended temp: {extras['temperature']:.4f}")
    
    print(f"\nCapabilities:")
    for k, v in extras['capabilities'].items():
        print(f"  {k}: {v:.4f}")
    print(f"Strengths: {extras['strengths']}")
    print(f"Weaknesses: {extras['weaknesses']}")
    
    # Test introspection
    features = torch.randn(2, 256)
    intro = limb.introspect(features)
    print(f"\nIntrospection results:")
    print(f"  Recommended strategy: {intro['recommended_strategy']}")
    print(f"  Should explore: {limb.should_explore(features)}")
    
    # Stats
    stats = limb.get_stats()
    print(f"\nMetaCognition Limb stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Parameter count
    total_params = sum(p.numel() for p in limb.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nMetaCognitionLimb tests passed!")
