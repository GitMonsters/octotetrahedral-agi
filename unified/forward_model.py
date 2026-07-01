"""
Unified Forward Model - Complete Cognitive Pipeline
====================================================

Orchestrates the complete unified cognitive stack:

1. Input Perception → Tetrahedral encoding
2. RNA Editing → Adaptive limb gating
3. Limbs Orchestrator → 8 limbs process in parallel
4. Quantum Coupling → Entanglement synchronization
5. Compound Reasoning → Multi-step inference
6. Action Generation → Output production

This is the main entry point for all inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import time
import math

from unified.limbs_orchestrator import UnifiedLimbsOrchestrator
from unified.rna_editing_layer import RNAEditingLayer, compute_rna_editing_loss
from unified.quantum_coupling import QuantumEntanglementLayer, compute_entanglement_entropy


class UnifiedForwardModel(nn.Module):
    """
    Complete unified cognitive pipeline.
    
    Flow:
        Input (tokens)
          ↓
        Perception Encoding (tetrahedral)
          ↓
        RNA Editing Layer (adaptive gating)
          ↓
        Limbs Orchestrator (8 limbs parallel)
          ↓
        Quantum Entanglement (coherent coupling)
          ↓
        Compound Reasoning (multi-step)
          ↓
        Output Generation
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 512,
        num_limbs: int = 8,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 2048,
        enable_quantum: bool = True,
        enable_rna_editing: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_limbs = num_limbs
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.enable_quantum = enable_quantum
        self.enable_rna_editing = enable_rna_editing
        
        # ════════════════════════════════════════════════════════════════
        # PERCEPTION: Input encoding
        # ════════════════════════════════════════════════════════════════
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.perception_norm = nn.LayerNorm(hidden_dim)
        
        # Tetrahedral projection (geometric embedding)
        self.tet_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # ════════════════════════════════════════════════════════════════
        # CORE COGNITIVE COMPONENTS
        # ════════════════════════════════════════════════════════════════
        
        # RNA Editing (adaptive limb gating)
        if enable_rna_editing:
            self.rna_editing = RNAEditingLayer(
                hidden_dim=hidden_dim,
                num_limbs=num_limbs,
                num_heads=num_heads,
                num_pathways=3,
                temperature_init=1.0
            )
        else:
            self.rna_editing = None
        
        # Limbs Orchestrator (8 cognitive limbs)
        self.limbs = UnifiedLimbsOrchestrator(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_limbs=num_limbs,
            enable_quantum_sync=enable_quantum
        )
        
        # Quantum Entanglement (coherent coupling)
        if enable_quantum:
            self.quantum_entanglement = QuantumEntanglementLayer(
                hidden_dim=hidden_dim,
                num_limbs=num_limbs,
                num_qubits=16,
                coupling_strength=0.1
            )
        else:
            self.quantum_entanglement = None
        
        # ════════════════════════════════════════════════════════════════
        # COMPOUND REASONING: Multi-step inference
        # ════════════════════════════════════════════════════════════════
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # ════════════════════════════════════════════════════════════════
        # ACTION GENERATION: Output production
        # ════════════════════════════════════════════════════════════════
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # ════════════════════════════════════════════════════════════════
        # AUXILIARY LOSSES & METRICS
        # ════════════════════════════════════════════════════════════════
        self.forward_count = 0
        self.last_metrics = {}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass through unified cognitive pipeline.
        
        Args:
            input_ids: [batch, seq_len] - token IDs
            attention_mask: Optional [batch, seq_len] - attention mask
            labels: Optional [batch, seq_len] - target tokens for loss
            return_all_layers: Whether to return intermediate activations
            
        Returns:
            Dict with:
                - logits: [batch, seq_len, vocab_size]
                - loss: Scalar loss (if labels provided)
                - metrics: Dict of metrics
                - intermediate_outputs: Optional dict of intermediate states
        """
        self.forward_count += 1
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: PERCEPTION - Input Encoding
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        
        # Token embedding
        x = self.embedding(input_ids)  # [batch, seq_len, hidden_dim]
        
        # Positional embedding
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        
        # Combine
        x = x + pos_emb
        x = self.perception_norm(x)
        
        # Tetrahedral projection
        x = self.tet_projection(x)
        x = F.gelu(x)
        
        perception_time = time.time() - t0
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: RNA EDITING - Adaptive Limb Gating
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        rna_info = None
        rna_loss = torch.tensor(0.0, device=device)
        
        if self.rna_editing is not None:
            rna_info = self.rna_editing(x, return_info=True)
            x_edited = rna_info['output']
            rna_loss = compute_rna_editing_loss(rna_info)
        else:
            x_edited = x
        
        rna_time = time.time() - t0
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: LIMBS ORCHESTRATOR - Parallel Processing
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        
        # Get RNA gates for limb routing
        rna_gates = rna_info['limb_gates'] if rna_info is not None else None
        
        limbs_output = self.limbs(
            x_edited,
            attention_mask=attention_mask,
            return_confidence=True,
            rna_gates=rna_gates
        )
        
        x_limbs = limbs_output['output']  # [batch, seq_len, hidden_dim]
        limb_weights = limbs_output['limb_weights']  # [batch, num_limbs]
        
        limbs_time = time.time() - t0
        
        # ════════════════════════════════════════════════════════════════
        # STEP 4: QUANTUM ENTANGLEMENT - Coherent Coupling
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        quantum_info = {}
        quantum_loss = torch.tensor(0.0, device=device)
        
        if self.enable_quantum and self.quantum_entanglement is not None:
            # Reconstruct limb states for quantum processing
            # [batch, seq_len, num_limbs, hidden_dim]
            limb_states = limbs_output['synchronized_outputs']
            
            quantum_result = self.quantum_entanglement(limb_states, attention_mask)
            x_quantum = quantum_result['output']
            
            quantum_info = {
                'entanglement_strength': quantum_result['entanglement_strength'],
                'coherence': quantum_result['coherence'],
            }
            
            # Quantum loss: encourage high entanglement
            # (more entanglement = better limb coupling)
            quantum_loss = -0.01 * quantum_result['entanglement_strength']
            
            x = x_limbs + 0.3 * x_quantum  # Blend
        else:
            x = x_limbs
        
        quantum_time = time.time() - t0
        
        # ════════════════════════════════════════════════════════════════
        # STEP 5: COMPOUND REASONING - Multi-step Inference
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        
        # Apply transformer reasoning layers
        for i, layer in enumerate(self.reasoning_layers):
            x = layer(x, src_key_padding_mask=attention_mask)
        
        reasoning_time = time.time() - t0
        
        # ════════════════════════════════════════════════════════════════
        # STEP 6: ACTION GENERATION - Output Production
        # ════════════════════════════════════════════════════════════════
        t0 = time.time()
        
        x = self.output_norm(x)
        logits = self.output_projection(x)  # [batch, seq_len, vocab_size]
        
        action_time = time.time() - t0
        
        # ════════════════════════════════════════════════════════════════
        # COMPUTE LOSS
        # ════════════════════════════════════════════════════════════════
        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )
            loss = ce_loss + rna_loss + quantum_loss
        
        # ════════════════════════════════════════════════════════════════
        # BUILD METRICS
        # ════════════════════════════════════════════════════════════════
        metrics = {
            'perception_time': perception_time,
            'rna_time': rna_time,
            'limbs_time': limbs_time,
            'quantum_time': quantum_time,
            'reasoning_time': reasoning_time,
            'action_time': action_time,
            'total_time': perception_time + rna_time + limbs_time + quantum_time + reasoning_time + action_time,
            'rna_loss': rna_loss.item() if rna_loss is not None else 0.0,
            'quantum_loss': quantum_loss.item() if quantum_loss is not None else 0.0,
        }
        
        if rna_info is not None:
            metrics['rna_confidence'] = rna_info['confidence'].mean().item()
            metrics['rna_temperature'] = rna_info['temperature'].mean().item() if isinstance(rna_info['temperature'], torch.Tensor) else rna_info['temperature']
        
        if quantum_info:
            metrics['entanglement_strength'] = quantum_info['entanglement_strength'].item()
            metrics['coherence'] = quantum_info['coherence'].item()
        
        if limbs_output.get('limb_confidences') is not None:
            confidences = limbs_output['limb_confidences']
            if confidences.dim() > 1:
                confidences = confidences.mean(dim=0)
            metrics['limb_confidences'] = confidences.detach().cpu().tolist()
        
        self.last_metrics = metrics
        
        # ════════════════════════════════════════════════════════════════
        # BUILD OUTPUT
        # ════════════════════════════════════════════════════════════════
        output = {
            'logits': logits,
            'loss': loss,
            'metrics': metrics,
        }
        
        if return_all_layers:
            output['intermediate_outputs'] = {
                'perception': x_edited if self.rna_editing is not None else x,
                'limbs': x_limbs,
                'quantum': x_quantum if self.enable_quantum else None,
                'reasoning': x,
            }
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            input_ids: [batch, seq_len] - initial tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            
        Returns:
            [batch, seq_len + max_new_tokens] - generated tokens
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate context if needed
                if generated.size(1) > self.max_seq_len:
                    context = generated[:, -self.max_seq_len:]
                else:
                    context = generated
                
                # Forward pass
                output = self.forward(context)
                logits = output['logits']
                
                # Get next token logits
                next_logits = logits[:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k is not None:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[:, indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'forward_count': self.forward_count,
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'num_limbs': self.num_limbs,
            'num_layers': self.num_layers,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'last_metrics': self.last_metrics,
            'quantum_enabled': self.enable_quantum,
            'rna_editing_enabled': self.enable_rna_editing,
        }


if __name__ == "__main__":
    print("Testing Unified Forward Model...")
    
    model = UnifiedForwardModel(
        vocab_size=1000,
        hidden_dim=256,
        num_limbs=8,
        num_heads=4,
        num_layers=3,
        enable_quantum=True,
        enable_rna_editing=True,
    )
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    print("\nForward pass with labels...")
    output = model(input_ids, labels=labels, return_all_layers=True)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Metrics: {output['metrics']}")
    
    print("\nGeneration test...")
    prompt = torch.randint(0, 1000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    
    print("\nModel stats...")
    stats = model.get_stats()
    print(f"Total params: {stats['total_params']:,}")
    print(f"Trainable params: {stats['trainable_params']:,}")
    
    print("\n✓ Unified Forward Model test passed!")
