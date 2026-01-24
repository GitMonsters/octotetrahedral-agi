"""
Perception Limb - Input encoding and embedding
Inspired by octopus arm chemoreceptors and mechanoreceptors

Biological insight:
- Octopus arms have ~10,000 chemoreceptors per sucker
- Arms can "taste" and "feel" simultaneously
- Local processing filters and encodes sensory data before sending to brain
- Proprioception provides arm position awareness

Our implementation:
- Token embedding (vocabulary -> hidden space)
- Positional encoding (sequence position awareness)
- Initial feature extraction layers
- Sensory normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .base_limb import BaseLimb


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with learnable scaling.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create sinusoidal encoding
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * 
            (-math.log(10000.0) / hidden_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, hidden_dim]
        
        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return self.dropout(x + self.scale * self.pe[:, :seq_len, :])


class PerceptionLimb(BaseLimb):
    """
    Perception Limb for input encoding.
    
    Transforms raw token IDs into rich hidden representations:
    1. Token embedding lookup
    2. Positional encoding
    3. Feature extraction through MLP
    4. Layer normalization
    """
    
    def __init__(
        self,
        vocab_size: int = 100277,
        hidden_dim: int = 256,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        buffer_size: int = 100
    ):
        # Initialize base (input_dim=hidden_dim since we embed first)
        super().__init__(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            buffer_size=buffer_size,
            limb_name="perception"
        )
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Feature extraction MLP (sensory processing)
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Input statistics tracking (for confidence estimation)
        self.register_buffer('_input_mean', torch.zeros(hidden_dim))
        self.register_buffer('_input_var', torch.ones(hidden_dim))
        self._input_count = 0
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with scaled normal distribution"""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
    
    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings with positional encoding.
        
        Args:
            token_ids: Token ID tensor [batch, seq_len]
            
        Returns:
            Embedded tensor [batch, seq_len, hidden_dim]
        """
        # Token embedding
        embeddings = self.token_embedding(token_ids)
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        
        return embeddings
    
    def process(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Process embedded input through feature extraction.
        
        Args:
            x: Embedded input [batch, seq_len, hidden_dim]
            
        Returns:
            Processed features [batch, seq_len, hidden_dim]
        """
        # Feature extraction with residual connection
        features = self.feature_mlp(x) + x
        
        # Normalize
        output = self.layer_norm(features)
        
        return output
    
    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        return_confidence: bool = False,
        update_stats: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Forward pass: token IDs -> embedded features.
        
        Args:
            token_ids: Input token IDs [batch, seq_len]
            embeddings: Pre-computed embeddings (alternative to token_ids)
            return_confidence: Whether to return confidence score
            update_stats: Whether to update input statistics
            
        Returns:
            Tuple of (output_features, confidence)
        """
        # Get embeddings
        if embeddings is not None:
            x = embeddings
        elif token_ids is not None:
            x = self.embed_tokens(token_ids)
        else:
            raise ValueError("Either token_ids or embeddings must be provided")
        
        # Update input statistics (for anomaly detection)
        if update_stats and self.training:
            self._update_input_stats(x)
        
        # Base transformation + LoRA adaptation
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        
        # Process through feature MLP
        output = self.process(adapted, **kwargs)
        
        if return_confidence:
            confidence = self.estimate_confidence(x, output)
            return output, confidence
        
        return output, None
    
    def _update_input_stats(self, x: torch.Tensor):
        """Update running statistics of input for anomaly detection"""
        with torch.no_grad():
            batch_mean = x.mean(dim=(0, 1))
            batch_var = x.var(dim=(0, 1))
            
            # Welford's online algorithm
            self._input_count += 1
            delta = batch_mean - self._input_mean
            self._input_mean = self._input_mean + delta / self._input_count
            self._input_var = (
                self._input_var * (self._input_count - 1) + 
                batch_var + 
                delta ** 2 * (self._input_count - 1) / self._input_count
            ) / self._input_count
    
    def estimate_confidence(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor
    ) -> float:
        """
        Estimate confidence based on input distribution match.
        
        Inputs far from training distribution get lower confidence.
        """
        base_confidence = super().estimate_confidence(input_tensor, output_tensor)
        
        # Add distribution-based confidence
        if self._input_count > 10:
            with torch.no_grad():
                input_mean = input_tensor.mean(dim=(0, 1))
                
                # Mahalanobis-like distance (simplified)
                diff = (input_mean - self._input_mean)
                distance = (diff ** 2 / (self._input_var + 1e-10)).mean()
                
                # Convert distance to confidence (sigmoid-like)
                dist_confidence = 1.0 / (1.0 + distance.item())
                
                # Blend with base confidence
                return 0.7 * base_confidence + 0.3 * dist_confidence
        
        return base_confidence
    
    def get_embedding_weight(self) -> torch.Tensor:
        """Get token embedding weight matrix"""
        return self.token_embedding.weight
    
    def get_stats(self) -> Dict[str, Any]:
        """Get limb statistics including embedding stats"""
        stats = super().get_stats()
        stats.update({
            'vocab_size': self.vocab_size,
            'embedding_norm': self.token_embedding.weight.norm().item(),
            'input_samples_seen': self._input_count,
            'pos_encoding_scale': self.pos_encoding.scale.item()
        })
        return stats


if __name__ == "__main__":
    print("Testing PerceptionLimb...")
    
    # Create limb
    limb = PerceptionLimb(
        vocab_size=100277,
        hidden_dim=256,
        max_seq_len=512
    )
    
    # Test with token IDs
    batch_size = 2
    seq_len = 20
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    output, confidence = limb(token_ids, return_confidence=True)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Confidence: {confidence:.4f}")
    
    # Test with pre-computed embeddings
    embeddings = torch.randn(batch_size, seq_len, 256)
    output2, _ = limb(embeddings=embeddings)
    print(f"Output from embeddings: {output2.shape}")
    
    # Test stats
    stats = limb.get_stats()
    print(f"\nLimb stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Parameter count
    total_params = sum(p.numel() for p in limb.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nPerceptionLimb tests passed!")
