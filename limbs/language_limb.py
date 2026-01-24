"""
Language Limb - Natural language understanding and generation
Inspired by octopus communication through color/texture patterns

Biological insight:
- Octopuses communicate through chromatophores (color cells)
- Can produce complex visual patterns for signaling
- Show context-dependent communication
- Demonstrate symbolic-like referencing

Our implementation:
- Token/subword processing
- Semantic embedding alignment
- Language-vision grounding
- Compositional understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

from .base_limb import BaseLimb


class LanguageAttention(nn.Module):
    """
    Causal self-attention for language processing.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len))
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal = self.causal_mask[:seq_len, :seq_len]
        attn = attn.masked_fill(causal == 0, float('-inf'))
        
        if attention_mask is not None:
            attn = attn.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        return out, attn_weights.mean(dim=1)


class SemanticGrounding(nn.Module):
    """
    Ground language representations in perceptual features.
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Project language to grounding space
        self.language_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Project perception to grounding space
        self.perception_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Grounding attention
        self.grounding_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output gate
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        language_features: torch.Tensor,
        perception_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Ground language in perception.
        
        Args:
            language_features: [batch, seq_len, hidden_dim]
            perception_features: [batch, num_percepts, hidden_dim]
            
        Returns:
            Grounded language features
        """
        if perception_features is None:
            return language_features
        
        # Project both
        lang_proj = self.language_proj(language_features)
        perc_proj = self.perception_proj(perception_features)
        
        # Cross-attention: language queries perception
        grounded, _ = self.grounding_attn(
            lang_proj, perc_proj, perc_proj
        )
        
        # Gate
        combined = torch.cat([language_features, grounded], dim=-1)
        gate = self.gate(combined)
        
        return gate * language_features + (1 - gate) * grounded


class LanguageLimb(BaseLimb):
    """
    Language Limb for natural language understanding and generation.
    
    Capabilities:
    1. Causal language modeling
    2. Semantic grounding in perception
    3. Compositional understanding
    4. Text generation
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        vocab_size: int = 50257,  # GPT-2 vocab size
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 512,
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
            limb_name="language"
        )
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Position embedding
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        
        # Language attention layers
        self.attention_layers = nn.ModuleList([
            LanguageAttention(hidden_dim, num_heads, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # FFN layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Semantic grounding
        self.grounding = SemanticGrounding(hidden_dim)
        
        # Output head (token prediction)
        self.output_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie weights with embedding
        self.output_head.weight = self.token_embed.weight
        
        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Stats
        self._tokens_processed = 0
        self._generations = 0
    
    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs to hidden representations."""
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        tok_emb = self.token_embed(token_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.pos_embed(positions)
        
        return tok_emb + pos_emb
    
    def process(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        perception_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Process language features through transformer layers.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
            attention_mask: Optional mask
            perception_features: Optional perception for grounding
            
        Returns:
            Processed features [batch, seq_len, hidden_dim]
        """
        # Transformer layers
        for i, (attn, ffn, attn_norm, ffn_norm) in enumerate(zip(
            self.attention_layers,
            self.ffn_layers,
            self.attn_norms,
            self.ffn_norms
        )):
            # Self-attention with residual
            attn_out, _ = attn(x, attention_mask)
            x = attn_norm(x + attn_out)
            
            # FFN with residual
            ffn_out = ffn(x)
            x = ffn_norm(x + ffn_out)
        
        # Ground in perception if available
        x = self.grounding(x, perception_features)
        
        # Final norm
        x = self.final_norm(x)
        
        self._tokens_processed += x.size(0) * x.size(1)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        perception_features: Optional[torch.Tensor] = None,
        return_confidence: bool = False,
        return_logits: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor]]:
        """
        Forward pass through language limb.
        
        Args:
            x: Pre-computed embeddings [batch, seq_len, hidden_dim]
               OR ignored if token_ids provided
            token_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            perception_features: Optional perception for grounding
            return_confidence: Whether to return confidence
            return_logits: Whether to return vocab logits
        """
        # Embed tokens if provided
        if token_ids is not None:
            x = self.embed_tokens(token_ids)
        
        # Base transformation + LoRA
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        
        # Language processing
        output = self.process(
            adapted,
            attention_mask=attention_mask,
            perception_features=perception_features,
            **kwargs
        )
        
        # Confidence
        confidence = None
        if return_confidence:
            confidence = self.estimate_confidence(x, output)
        
        # Token logits
        logits = None
        if return_logits:
            logits = self.output_head(output)
        
        return output, confidence, logits
    
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        perception_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            prompt_ids: Starting token IDs [batch, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            perception_features: Optional perception context
            
        Returns:
            Generated token IDs [batch, prompt_len + new_tokens]
        """
        self.eval()
        generated = prompt_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if needed
                if generated.size(1) > self.max_seq_len:
                    context = generated[:, -self.max_seq_len:]
                else:
                    context = generated
                
                # Forward pass
                output, _, logits = self(
                    x=None,
                    token_ids=context,
                    perception_features=perception_features,
                    return_logits=True
                )
                
                # Get next token logits
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
        
        self._generations += 1
        return generated
    
    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode text to semantic representation.
        
        Args:
            token_ids: [batch, seq_len]
            
        Returns:
            Semantic embedding [batch, hidden_dim]
        """
        output, _, _ = self(x=None, token_ids=token_ids)
        return output.mean(dim=1)  # Mean pooling
    
    def get_stats(self) -> Dict[str, Any]:
        """Get language limb statistics."""
        stats = super().get_stats()
        stats.update({
            'vocab_size': self.vocab_size,
            'tokens_processed': self._tokens_processed,
            'generations': self._generations
        })
        return stats


if __name__ == "__main__":
    print("Testing LanguageLimb...")
    
    # Create limb
    limb = LanguageLimb(
        hidden_dim=256,
        vocab_size=1000,  # Small vocab for testing
        num_heads=4,
        num_layers=2,
        max_seq_len=128
    )
    
    # Test input
    batch_size = 2
    seq_len = 20
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass with token IDs
    output, confidence, logits = limb(
        x=None,
        token_ids=token_ids,
        return_confidence=True,
        return_logits=True
    )
    
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Confidence: {confidence:.4f}")
    
    # Test with perception grounding
    perception = torch.randn(batch_size, 10, 256)
    output_grounded, _, _ = limb(
        x=None,
        token_ids=token_ids,
        perception_features=perception
    )
    print(f"Grounded output shape: {output_grounded.shape}")
    
    # Test generation
    prompt = torch.randint(0, 1000, (1, 5))
    generated = limb.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8
    )
    print(f"Generated shape: {generated.shape}")
    
    # Test encoding
    encoding = limb.encode_text(token_ids)
    print(f"Text encoding shape: {encoding.shape}")
    
    # Stats
    stats = limb.get_stats()
    print(f"\nLanguage Limb stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Parameter count
    total_params = sum(p.numel() for p in limb.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nLanguageLimb tests passed!")
