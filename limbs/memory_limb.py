"""
Memory Limb - Long-term and episodic memory processing
Inspired by octopus vertical lobe memory systems

Biological insight:
- Octopus vertical lobe is key for memory formation
- Can remember for months (long-term memory exists)
- Shows both associative and episodic-like memory
- Memory is distributed across multiple lobes

Our implementation:
- Differentiable key-value memory store
- Episodic buffer with temporal context
- Memory retrieval via attention
- Memory consolidation mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from collections import deque

from .base_limb import BaseLimb


class EpisodicMemory:
    """
    Episodic memory buffer storing experiences with temporal context.
    """
    
    def __init__(self, capacity: int = 1000, hidden_dim: int = 256):
        self.capacity = capacity
        self.hidden_dim = hidden_dim
        self.memories: deque = deque(maxlen=capacity)
        self.timestamps: deque = deque(maxlen=capacity)
        self.importance: deque = deque(maxlen=capacity)
        self.current_time = 0
    
    def store(
        self,
        memory: torch.Tensor,
        importance: float = 1.0,
        context: Optional[Dict] = None
    ):
        """Store a memory with timestamp and importance."""
        self.memories.append(memory.detach())  # stay on same device
        self.timestamps.append(self.current_time)
        self.importance.append(importance)
        self.current_time += 1
    
    def retrieve_recent(self, n: int = 10) -> List[torch.Tensor]:
        """Retrieve n most recent memories."""
        return list(self.memories)[-n:]
    
    def retrieve_by_importance(self, n: int = 10) -> List[torch.Tensor]:
        """Retrieve n most important memories."""
        if len(self.memories) == 0:
            return []
        
        indices = sorted(
            range(len(self.importance)),
            key=lambda i: self.importance[i],
            reverse=True
        )[:n]
        return [self.memories[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.memories)


class KeyValueMemory(nn.Module):
    """
    Differentiable key-value memory store.
    Used for semantic/long-term memory.
    """
    
    def __init__(
        self,
        num_slots: int = 128,
        key_dim: int = 64,
        value_dim: int = 256
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Memory slots
        self.keys = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(num_slots, value_dim) * 0.02)
        
        # Query projection
        self.query_proj = nn.Linear(value_dim, key_dim)
        
        # Usage tracking (not gradient)
        self.register_buffer('usage', torch.zeros(num_slots))
    
    def read(
        self,
        query: torch.Tensor,
        top_k: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using query.
        
        Args:
            query: Query tensor [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            top_k: Number of memory slots to attend to
            
        Returns:
            (retrieved_value, attention_weights)
        """
        # Handle different input shapes
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        batch_size, seq_len, _ = query.shape
        
        # Project query to key space
        q = self.query_proj(query)  # [batch, seq_len, key_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, self.keys.t())  # [batch, seq_len, num_slots]
        scores = scores / math.sqrt(self.key_dim)
        
        # Top-k sparse attention
        if top_k < self.num_slots:
            top_scores, top_indices = scores.topk(top_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, top_indices, 1.0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)  # [batch, seq_len, num_slots]
        
        # Read values
        retrieved = torch.matmul(weights, self.values)  # [batch, seq_len, value_dim]
        
        # Update usage (for memory management)
        with torch.no_grad():
            self.usage = 0.99 * self.usage + 0.01 * weights.sum(dim=[0, 1])
        
        return retrieved.squeeze(1), weights
    
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        strength: float = 0.1
    ):
        """
        Write to memory (update least-used slot).
        
        Args:
            key: Key to write [hidden_dim]
            value: Value to write [hidden_dim]
            strength: Write strength
        """
        # Find least-used slot
        idx = self.usage.argmin().item()
        
        # Project key
        k = self.query_proj(value)  # Use value for key projection
        
        # Soft write
        with torch.no_grad():
            self.keys.data[idx] = (1 - strength) * self.keys.data[idx] + strength * k
            self.values.data[idx] = (1 - strength) * self.values.data[idx] + strength * value
            self.usage[idx] = 1.0


class MemoryLimb(BaseLimb):
    """
    Memory Limb for long-term and episodic memory processing.
    
    Capabilities:
    1. Episodic memory storage and retrieval
    2. Semantic memory via key-value store
    3. Memory consolidation (compress frequent patterns)
    4. Temporal context encoding
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_memory_slots: int = 128,
        episodic_capacity: int = 1000,
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
            limb_name="memory"
        )
        
        self.hidden_dim = hidden_dim
        
        # Semantic memory (key-value store)
        self.semantic_memory = KeyValueMemory(
            num_slots=num_memory_slots,
            key_dim=hidden_dim // 4,
            value_dim=hidden_dim
        )
        
        # Episodic memory buffer
        self.episodic_memory = EpisodicMemory(
            capacity=episodic_capacity,
            hidden_dim=hidden_dim
        )
        
        # Memory retrieval attention
        self.retrieval_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Memory consolidation network
        self.consolidation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal position encoding for episodic context
        self.temporal_encoding = nn.Embedding(1000, hidden_dim)
        
        # Gate for blending retrieved vs current
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Stats tracking
        self._last_retrieval_weights = None
        self._memories_stored = 0
        self._retrievals_performed = 0
    
    def process(
        self,
        x: torch.Tensor,
        store_memory: bool = True,
        retrieve_memory: bool = True,
        importance: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Process input with memory operations.
        
        Args:
            x: Input features [batch, seq_len, hidden_dim]
            store_memory: Whether to store current input as episodic memory
            retrieve_memory: Whether to retrieve from semantic memory
            importance: Importance score for memory storage
            
        Returns:
            Memory-enhanced features [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Store to episodic memory if requested
        if store_memory:
            pooled = x.mean(dim=1)  # [batch, hidden_dim]
            for i in range(batch_size):
                self.episodic_memory.store(pooled[i], importance=importance)
                self._memories_stored += 1
        
        # Retrieve from semantic memory
        if retrieve_memory:
            query = x.mean(dim=1)  # Pool for query
            retrieved, weights = self.semantic_memory.read(query)
            self._last_retrieval_weights = weights.detach()
            self._retrievals_performed += 1
            
            # Expand retrieved to sequence length
            retrieved = retrieved.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Gate between current and retrieved
            gate_input = torch.cat([x, retrieved], dim=-1)
            gate = self.memory_gate(gate_input)
            
            # Blend
            x = gate * x + (1 - gate) * retrieved
        
        # Consolidation: compress and refine
        context = x.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        consolidated = self.consolidation_net(torch.cat([x, context], dim=-1))
        
        # Output
        output = self.norm(x + consolidated)
        output = self.output_proj(output)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        store_memory: bool = True,
        retrieve_memory: bool = True,
        return_confidence: bool = False,
        importance: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor]]:
        """
        Forward pass through memory limb.
        """
        # Base transformation + LoRA
        base_out = self.transform(x)
        lora_out = self.lora(x)
        adapted = base_out + lora_out
        
        # Memory processing
        output = self.process(
            adapted,
            store_memory=store_memory,
            retrieve_memory=retrieve_memory,
            importance=importance,
            **kwargs
        )
        
        # Confidence estimation
        confidence = None
        if return_confidence:
            confidence = self.estimate_confidence(x, output)
        
        return output, confidence, self._last_retrieval_weights
    
    def write_to_semantic(
        self,
        value: torch.Tensor,
        strength: float = 0.1
    ):
        """Explicitly write to semantic memory."""
        if value.dim() > 1:
            value = value.mean(dim=0)  # Pool to single vector
        self.semantic_memory.write(value, value, strength=strength)
    
    def consolidate_episodic_to_semantic(self, threshold: int = 10):
        """
        Consolidate frequently-accessed episodic memories to semantic memory.
        This simulates sleep-like memory consolidation.
        """
        if len(self.episodic_memory) < threshold:
            return
        
        # Get most important episodic memories
        important = self.episodic_memory.retrieve_by_importance(threshold)
        
        if len(important) > 0:
            # Average and write to semantic
            stacked = torch.stack(important)
            consolidated = stacked.mean(dim=0)
            self.write_to_semantic(consolidated, strength=0.05)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory limb statistics."""
        stats = super().get_stats()
        stats.update({
            'episodic_count': len(self.episodic_memory),
            'semantic_usage_mean': self.semantic_memory.usage.mean().item(),
            'semantic_usage_max': self.semantic_memory.usage.max().item(),
            'memories_stored': self._memories_stored,
            'retrievals_performed': self._retrievals_performed
        })
        return stats


if __name__ == "__main__":
    print("Testing MemoryLimb...")
    
    # Create limb
    limb = MemoryLimb(
        hidden_dim=256,
        num_memory_slots=64,
        episodic_capacity=100
    )
    
    # Test input
    batch_size = 2
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 256)
    
    # Forward pass
    output, confidence, retrieval_weights = limb(
        x,
        store_memory=True,
        retrieve_memory=True,
        return_confidence=True,
        importance=0.8
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Confidence: {confidence:.4f}")
    
    # Multiple iterations to build memory
    for i in range(10):
        x_new = torch.randn(batch_size, seq_len, 256)
        output, _, _ = limb(x_new, importance=float(i) / 10)
    
    # Test consolidation
    limb.consolidate_episodic_to_semantic(threshold=5)
    
    # Stats
    stats = limb.get_stats()
    print(f"\nMemory Limb stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Parameter count
    total_params = sum(p.numel() for p in limb.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nMemoryLimb tests passed!")
