"""
Action Limb - Output generation and motor control
Inspired by octopus arm motor programs

Biological insight:
- Octopus arms have 'motor programs' that execute complex movements
- Fetching movement: stereotyped reaching with stiffness propagation
- Arms can execute programs locally without brain involvement
- Multiple arms can coordinate for complex tasks

Our implementation:
- Project hidden state to vocabulary logits
- Temperature-controlled sampling
- Confidence-gated output
- Optional action repetition for consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

from .base_limb import BaseLimb


class ActionLimb(BaseLimb):
    """
    Action Limb for output generation.
    
    Transforms reasoning outputs into:
    1. Vocabulary logits for next token prediction
    2. Confidence scores for output gating
    3. Action probabilities for sampling
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        vocab_size: int = 100277,
        dropout: float = 0.1,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        buffer_size: int = 100,
        tie_weights: bool = True
    ):
        super().__init__(
            input_dim=hidden_dim,
            output_dim=hidden_dim,  # Keep hidden_dim for base transform
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            buffer_size=buffer_size,
            limb_name="action"
        )
        
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        # Pre-output transformation
        self.pre_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Confidence gate (decides whether to output or abstain)
        self.confidence_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Temperature parameter (learnable)
        self.log_temperature = nn.Parameter(torch.zeros(1))
        
        # Action repetition detection (for consistency)
        self._last_action = None
        self._repetition_count = 0
    
    @property
    def temperature(self) -> float:
        """Get current temperature (clamped to reasonable range)"""
        return torch.clamp(torch.exp(self.log_temperature), 0.1, 5.0).item()
    
    def process(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Process input through pre-output layers.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
            
        Returns:
            Processed features [batch, seq_len, hidden_dim]
        """
        return self.pre_output(x)
    
    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
        return_logits: bool = True,
        embedding_weight: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor]]:
        """
        Forward pass: hidden state -> vocabulary logits.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
            return_confidence: Whether to return confidence score
            return_logits: Whether to return raw logits (vs probabilities)
            embedding_weight: Optional embedding weight for tied projections
            
        Returns:
            Tuple of (logits/probs, confidence, gate_values)
        """
        # Base transformation + LoRA
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        
        # Pre-output processing
        hidden = self.process(adapted, **kwargs)
        
        # Output projection
        if self.tie_weights and embedding_weight is not None:
            # Tied embeddings: use transposed embedding weight
            logits = F.linear(hidden, embedding_weight)
        else:
            logits = self.output_projection(hidden)
        
        # Apply temperature
        logits = logits / self.temperature
        
        # Confidence gating
        gate_values = None
        confidence = None
        
        if return_confidence:
            gate_values = self.confidence_gate(hidden)  # [batch, seq_len, 1]
            confidence = gate_values.mean().item()
        
        if not return_logits:
            logits = F.softmax(logits, dim=-1)
        
        return logits, confidence, gate_values
    
    def sample(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample tokens from logits.
        
        Args:
            logits: Vocabulary logits [batch, seq_len, vocab_size]
            temperature: Sampling temperature (overrides learned)
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            
        Returns:
            Tuple of (sampled_tokens, log_probs)
        """
        # Apply temperature
        temp = temperature if temperature is not None else self.temperature
        logits = logits / temp
        
        # Top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter back
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
        tokens = tokens.view(probs.size(0), probs.size(1))
        
        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        
        return tokens, selected_log_probs
    
    def greedy_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Greedy decoding (argmax).
        
        Args:
            logits: Vocabulary logits [batch, seq_len, vocab_size]
            
        Returns:
            Token IDs [batch, seq_len]
        """
        return logits.argmax(dim=-1)
    
    def estimate_confidence(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor
    ) -> float:
        """
        Estimate output confidence using multiple signals.
        """
        base_conf = super().estimate_confidence(input_tensor, output_tensor)
        
        # Entropy-based confidence (lower entropy = higher confidence)
        with torch.no_grad():
            if output_tensor.dim() == 3:  # [batch, seq, vocab]
                probs = F.softmax(output_tensor, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
                max_entropy = math.log(output_tensor.size(-1))
                entropy_conf = 1.0 - (entropy / max_entropy).item()
            else:
                entropy_conf = 0.5
        
        # Peak probability confidence
        peak_conf = 0.5
        with torch.no_grad():
            if output_tensor.dim() == 3:
                probs = F.softmax(output_tensor, dim=-1)
                peak_conf = probs.max(dim=-1)[0].mean().item()
        
        return 0.3 * base_conf + 0.4 * entropy_conf + 0.3 * peak_conf
    
    def set_temperature(self, temperature: float):
        """Manually set temperature"""
        self.log_temperature.data = torch.log(torch.tensor([temperature]))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get limb statistics"""
        stats = super().get_stats()
        stats.update({
            'vocab_size': self.vocab_size,
            'temperature': self.temperature,
            'tie_weights': self.tie_weights,
            'output_proj_norm': (
                self.output_projection.weight.norm().item() 
                if self.output_projection is not None 
                else 0.0
            )
        })
        return stats
    
    def tie_embedding_weights(self, embedding_weight: torch.Tensor):
        """
        Tie output projection to embedding weights.
        
        Args:
            embedding_weight: Embedding weight matrix [vocab_size, hidden_dim]
        """
        if embedding_weight.shape != (self.vocab_size, self.transform.in_features):
            raise ValueError(
                f"Embedding weight shape {embedding_weight.shape} doesn't match "
                f"expected ({self.vocab_size}, {self.transform.in_features})"
            )
        
        # Delete existing projection and use embedding weight
        del self.output_projection
        self.output_projection = None
        self._tied_embedding = embedding_weight
        self.tie_weights = True


if __name__ == "__main__":
    print("Testing ActionLimb...")
    
    # Create limb
    limb = ActionLimb(
        hidden_dim=256,
        vocab_size=100277,
        dropout=0.1
    )
    
    # Test input
    batch_size = 2
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 256)
    
    # Forward pass
    logits, confidence, gate = limb(x, return_confidence=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Gate values shape: {gate.shape}")
    
    # Test sampling
    tokens, log_probs = limb.sample(logits, temperature=0.8, top_k=50)
    print(f"Sampled tokens shape: {tokens.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    
    # Test greedy
    greedy_tokens = limb.greedy_decode(logits)
    print(f"Greedy tokens shape: {greedy_tokens.shape}")
    
    # Test temperature
    print(f"\nCurrent temperature: {limb.temperature:.4f}")
    limb.set_temperature(0.5)
    print(f"After setting to 0.5: {limb.temperature:.4f}")
    
    # Stats
    stats = limb.get_stats()
    print(f"\nLimb stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Parameter count
    total_params = sum(p.numel() for p in limb.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nActionLimb tests passed!")
