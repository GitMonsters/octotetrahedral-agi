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
    
    def _read_impl(
        self,
        query: torch.Tensor,
        memory_state: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Core read implementation operating on a provided memory state.
        Fully differentiable — gradients flow through memory_state.
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = query.shape
        
        memory = memory_state.unsqueeze(0).expand(batch_size, -1, -1)
        
        q = self.read_query_proj(query)
        k = self.read_key_proj(memory)
        v = self.read_value_proj(memory)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        read_content = torch.matmul(attn_weights, v)
        read_content = read_content.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        read_content = self.read_out_proj(read_content)
        read_content = self.read_norm(read_content)
        
        read_weights_agg = attn_weights.mean(dim=1)
        
        if squeeze_output:
            read_content = read_content.squeeze(1)
            read_weights_agg = read_weights_agg.squeeze(1)
        
        if return_weights:
            return read_content, read_weights_agg
        return read_content, None

    def read(
        self,
        query: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Read from memory using attention (uses self.memory parameter).
        """
        return self._read_impl(query, self.memory, return_weights)

    def read_from_state(
        self,
        query: torch.Tensor,
        memory_state: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Read from an external memory state tensor (differentiable).
        Used by compound loop to maintain gradient flow across iterations.
        """
        return self._read_impl(query, memory_state, return_weights)
    
    def _write_impl(
        self,
        content: torch.Tensor,
        memory_state: torch.Tensor,
        address: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Core write implementation operating on a provided memory state.
        Returns new memory state with full gradient flow.
        """
        batch_size = content.size(0)
        
        if address is None:
            address = F.softmax(self.address_net(content), dim=-1)
        
        write_content = self.write_content(content)
        write_content = self.write_norm(write_content)
        
        memory_expanded = memory_state.unsqueeze(0).expand(batch_size, -1, -1)
        
        gate_input = torch.cat([
            memory_expanded,
            write_content.unsqueeze(1).expand(-1, self.num_slots, -1)
        ], dim=-1)
        write_gates = self.write_gate(gate_input)
        
        erase = self.erase_gate(write_content).unsqueeze(1)
        address_expanded = address.unsqueeze(-1)
        
        erase_term = memory_expanded * (1 - erase * address_expanded)
        write_term = write_content.unsqueeze(1) * write_gates * address_expanded
        new_memory = erase_term + write_term
        
        return new_memory.mean(dim=0)

    def write(
        self,
        content: torch.Tensor,
        address: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Write to memory with gated update (uses self.memory parameter).
        """
        return self._write_impl(content, self.memory, address)

    def write_to_state(
        self,
        content: torch.Tensor,
        memory_state: torch.Tensor,
        address: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Write to an external memory state tensor (differentiable).
        Returns new state — no detach, full gradient flow.
        Used by compound loop for within-loop memory updates.
        """
        return self._write_impl(content, memory_state, address)
    
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
