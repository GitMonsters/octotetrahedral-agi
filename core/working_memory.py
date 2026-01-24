"""
Working Memory Module
Differentiable working memory with read/write operations

Inspired by:
- Neural Turing Machines (Graves et al.)
- Differentiable Neural Computers
- Octopus distributed memory (local in arms, global in brain)

The working memory provides:
1. Persistent storage across processing steps
2. Attention-based read operations
3. Gated write operations
4. Semantic slot organization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class WorkingMemory(nn.Module):
    """
    Differentiable working memory with attention-based access.
    
    Memory is organized into semantic slots:
    - Slot 0: Goal/task representation
    - Slot 1: Current context
    - Slot 2: Intermediate results
    - Slot 3: Output buffer
    
    These semantics are soft (learned), not hard-coded.
    """
    
    def __init__(
        self,
        num_slots: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Learnable memory content (initialized randomly, learned during training)
        self.memory = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        
        # Read mechanism: multi-head attention
        self.read_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.read_key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.read_value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.read_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Write mechanism: gated update
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.write_content = nn.Linear(hidden_dim, hidden_dim)
        
        # Erase mechanism (for selective forgetting)
        self.erase_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Slot addressing (which slot to write to)
        self.address_net = nn.Linear(hidden_dim, num_slots)
        
        # Layer norms
        self.read_norm = nn.LayerNorm(hidden_dim)
        self.write_norm = nn.LayerNorm(hidden_dim)
    
    def read(
        self,
        query: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Read from memory using attention.
        
        Args:
            query: Query tensor [batch, hidden_dim] or [batch, seq_len, hidden_dim]
            return_weights: Whether to return attention weights
            
        Returns:
            read_content: Content read from memory [batch, hidden_dim] or [batch, seq_len, hidden_dim]
            read_weights: Optional attention weights [batch, num_slots] or [batch, seq_len, num_slots]
        """
        # Handle different query shapes
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, hidden]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = query.shape
        
        # Expand memory for batch
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, slots, hidden]
        
        # Project query, key, value
        q = self.read_query_proj(query)  # [batch, seq, hidden]
        k = self.read_key_proj(memory)   # [batch, slots, hidden]
        v = self.read_value_proj(memory) # [batch, slots, hidden]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, heads, seq, slots]
        
        # Weighted sum
        read_content = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
        read_content = read_content.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        read_content = self.read_out_proj(read_content)
        read_content = self.read_norm(read_content)
        
        # Aggregate weights across heads
        read_weights_agg = attn_weights.mean(dim=1)  # [batch, seq, slots]
        
        if squeeze_output:
            read_content = read_content.squeeze(1)
            read_weights_agg = read_weights_agg.squeeze(1)
        
        if return_weights:
            return read_content, read_weights_agg
        return read_content, None
    
    def write(
        self,
        content: torch.Tensor,
        address: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Write to memory with gated update.
        
        Args:
            content: Content to write [batch, hidden_dim]
            address: Optional explicit address weights [batch, num_slots]
                     If None, computed from content
                     
        Returns:
            Updated memory [num_slots, hidden_dim]
        """
        batch_size = content.size(0)
        
        # Compute address if not provided
        if address is None:
            address = F.softmax(self.address_net(content), dim=-1)  # [batch, slots]
        
        # Compute write content
        write_content = self.write_content(content)  # [batch, hidden]
        write_content = self.write_norm(write_content)
        
        # Expand for all slots
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, slots, hidden]
        
        # Compute write gate (how much to update)
        # Concatenate current memory and new content for each slot
        gate_input = torch.cat([
            memory_expanded,
            write_content.unsqueeze(1).expand(-1, self.num_slots, -1)
        ], dim=-1)  # [batch, slots, hidden*2]
        write_gates = self.write_gate(gate_input)  # [batch, slots, hidden]
        
        # Compute erase gate
        erase = self.erase_gate(write_content).unsqueeze(1)  # [batch, 1, hidden]
        
        # Address-weighted update
        address_expanded = address.unsqueeze(-1)  # [batch, slots, 1]
        
        # New memory content per batch
        # memory_new = memory * (1 - erase * address) + write_content * write_gate * address
        erase_term = memory_expanded * (1 - erase * address_expanded)
        write_term = write_content.unsqueeze(1) * write_gates * address_expanded
        new_memory = erase_term + write_term  # [batch, slots, hidden]
        
        # Average across batch for parameter update
        # In practice, we update self.memory based on the batch
        # For simplicity, we return the computed memory and let the caller decide
        return new_memory.mean(dim=0)  # [slots, hidden]
    
    def forward(
        self,
        query: torch.Tensor,
        write_content: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Combined read and optional write operation.
        
        Args:
            query: Query for reading [batch, hidden_dim] or [batch, seq_len, hidden_dim]
            write_content: Optional content to write [batch, hidden_dim]
            
        Returns:
            Dictionary containing:
                - 'read_content': Content read from memory
                - 'read_weights': Attention weights over slots
                - 'memory_state': Current memory state
        """
        # Read from memory
        read_content, read_weights = self.read(query, return_weights=True)
        
        # Optionally write to memory
        if write_content is not None:
            new_memory = self.write(write_content)
            # Update memory parameter (in-place would require no_grad context)
            # For forward pass, we just compute what it would be
            self.memory.data = new_memory.detach()
        
        return {
            'read_content': read_content,
            'read_weights': read_weights,
            'memory_state': self.memory.clone()
        }
    
    def reset(self):
        """Reset memory to initial random state"""
        nn.init.normal_(self.memory, std=0.02)
    
    def get_memory_state(self) -> torch.Tensor:
        """Return current memory state"""
        return self.memory.clone()


if __name__ == "__main__":
    # Test working memory
    print("Testing WorkingMemory...")
    
    batch_size = 2
    hidden_dim = 256
    num_slots = 4
    
    memory = WorkingMemory(num_slots=num_slots, hidden_dim=hidden_dim)
    
    # Test read
    query = torch.randn(batch_size, hidden_dim)
    read_content, read_weights = memory.read(query, return_weights=True)
    print(f"Read content shape: {read_content.shape}")
    print(f"Read weights shape: {read_weights.shape}")
    print(f"Read weights sum: {read_weights.sum(dim=-1)}")  # Should be 1
    
    # Test read with sequence
    query_seq = torch.randn(batch_size, 10, hidden_dim)
    read_content_seq, read_weights_seq = memory.read(query_seq, return_weights=True)
    print(f"Sequence read content shape: {read_content_seq.shape}")
    print(f"Sequence read weights shape: {read_weights_seq.shape}")
    
    # Test write
    write_content = torch.randn(batch_size, hidden_dim)
    new_memory = memory.write(write_content)
    print(f"New memory shape: {new_memory.shape}")
    
    # Test forward (combined)
    result = memory(query, write_content=write_content)
    print(f"Forward read content shape: {result['read_content'].shape}")
    print(f"Memory state shape: {result['memory_state'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in memory.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\nAll memory tests passed!")
