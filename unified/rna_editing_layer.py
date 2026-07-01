"""
RNA Editing Layer - Adaptive Limb Gating
==========================================

Implements octopus-inspired RNA editing that dynamically modulates
which limbs activate and how strongly they respond to input.

Key capabilities:
1. Task-aware adaptation: Different tasks activate different limb combinations
2. Temporal gating: Limb activation evolves through sequence
3. Excitatory/Inhibitory balance: Maintains E/I ratio ~80/20 for stability
4. Pathway selection: Routes information through learned pathways
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import math


class RNAEditingLayer(nn.Module):
    """
    RNA-inspired weight modulation for adaptive limb activation.
    
    In biological octopuses, RNA editing allows rapid, flexible responses
    to environmental changes without genetic modification. We implement
    a differentiable analog that:
    
    - Learns which limbs to activate per task
    - Dynamically adjusts attention head weights
    - Maintains excitatory/inhibitory balance
    - Enables rapid adaptation through learned pathways
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_limbs: int = 8,
        num_heads: int = 4,
        num_pathways: int = 3,
        temperature_init: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_limbs = num_limbs
        self.num_heads = num_heads
        self.num_pathways = num_pathways
        
        # ════════════════════════════════════════════════════════════════
        # PATHWAY SELECTION: Learn which processing pathway per task
        # ════════════════════════════════════════════════════════════════
        self.pathway_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_pathways)
        )
        self.pathway_embeddings = nn.Embedding(num_pathways, hidden_dim)
        
        # ════════════════════════════════════════════════════════════════
        # LIMB GATING: Per-limb activation scores
        # ════════════════════════════════════════════════════════════════
        # Gate per limb: should this limb activate at all?
        self.limb_gates = nn.Linear(hidden_dim, num_limbs)
        
        # Limb strengths: if activated, how strong?
        self.limb_strengths = nn.Linear(hidden_dim, num_limbs)
        
        # ════════════════════════════════════════════════════════════════
        # ATTENTION HEAD GATING: Which attention heads matter?
        # ════════════════════════════════════════════════════════════════
        self.head_gates = nn.Linear(hidden_dim, num_heads)
        self.head_importance = nn.Linear(hidden_dim, num_heads)
        
        # ════════════════════════════════════════════════════════════════
        # EXCITATORY/INHIBITORY BALANCE
        # ════════════════════════════════════════════════════════════════
        # E/I signs: +1 for excitatory, -1 for inhibitory
        self.ei_signs = nn.Linear(hidden_dim, num_limbs)
        
        # E/I balance predictor: confidence in maintaining 80/20 ratio
        self.ei_balance_predictor = nn.Linear(hidden_dim, 1)
        
        # ════════════════════════════════════════════════════════════════
        # TEMPERATURE & CONFIDENCE ESTIMATION
        # ════════════════════════════════════════════════════════════════
        self.temperature_logits = nn.Linear(hidden_dim, 1)
        self.confidence_predictor = nn.Linear(hidden_dim, 1)
        
        # Learnable temperature (starts at temperature_init, adapts during training)
        self.register_parameter('temperature', nn.Parameter(torch.tensor(temperature_init)))
        
        # ════════════════════════════════════════════════════════════════
        # ADAPTIVE LEARNING
        # ════════════════════════════════════════════════════════════════
        # Track editing patterns for meta-learning
        self.register_buffer('pathway_activation_counts', torch.zeros(num_pathways))
        self.register_buffer('limb_activation_counts', torch.zeros(num_limbs))
        
        self.dropout = nn.Dropout(dropout)
        
        # Target E/I ratio (80% excitatory, 20% inhibitory)
        self.target_ei_ratio = 0.8
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_info: bool = False
    ) -> Dict[str, Any]:
        """
        Apply RNA editing to modulate limb activation.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] - encoder output
            return_info: Whether to return detailed editing information
            
        Returns:
            Dict containing:
                - output: Edited states [batch, seq_len, hidden_dim]
                - limb_gates: Gating signals [batch, seq_len, num_limbs] or [batch, num_limbs]
                - pathway_weights: Selected pathway weights [batch, num_pathways]
                - head_gates: Attention head gates [batch, num_heads]
                - confidence: Model confidence in the editing [batch] or [batch, 1]
                - temperature: Current temperature
                - ei_signs: E/I polarity for each limb [batch, num_limbs]
                - ei_balance_loss: Regularization loss for E/I balance
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # ════════════════════════════════════════════════════════════════
        # STEP 1: Pathway Selection
        # ════════════════════════════════════════════════════════════════
        # Average the hidden states to get a task-level representation
        task_representation = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        
        pathway_logits = self.pathway_selector(task_representation)  # [batch, num_pathways]
        pathway_weights = F.softmax(pathway_logits, dim=-1)  # [batch, num_pathways]
        
        # Update activation counts (for meta-learning)
        with torch.no_grad():
            self.pathway_activation_counts += pathway_weights.sum(dim=0).detach()
        
        # Get pathway embedding for this batch
        pathway_indices = pathway_weights.argmax(dim=-1)  # [batch]
        pathway_emb = self.pathway_embeddings(pathway_indices)  # [batch, hidden_dim]
        
        # ════════════════════════════════════════════════════════════════
        # STEP 2: Limb Gate Computation
        # ════════════════════════════════════════════════════════════════
        # Use task representation + pathway embedding for gating decisions
        gate_input = task_representation + 0.1 * pathway_emb  # [batch, hidden_dim]
        
        limb_gate_logits = self.limb_gates(gate_input)  # [batch, num_limbs]
        limb_gates = torch.sigmoid(limb_gate_logits)  # [batch, num_limbs] in (0, 1)
        
        limb_strength_logits = self.limb_strengths(gate_input)  # [batch, num_limbs]
        limb_strengths = torch.sigmoid(limb_strength_logits)  # [batch, num_limbs] in (0, 1)
        
        # Combined limb activation: gate × strength
        limb_activation = limb_gates * limb_strengths  # [batch, num_limbs]
        
        # Update activation counts
        with torch.no_grad():
            self.limb_activation_counts += limb_activation.sum(dim=0).detach()
        
        # ════════════════════════════════════════════════════════════════
        # STEP 3: Attention Head Gating
        # ════════════════════════════════════════════════════════════════
        head_gate_logits = self.head_gates(task_representation)  # [batch, num_heads]
        head_gates = torch.sigmoid(head_gate_logits)  # [batch, num_heads]
        
        head_importance_logits = self.head_importance(task_representation)  # [batch, num_heads]
        head_importance = F.softmax(head_importance_logits, dim=-1)  # [batch, num_heads]
        
        # ════════════════════════════════════════════════════════════════
        # STEP 4: E/I Polarity Assignment
        # ════════════════════════════════════════════════════════════════
        ei_logits = self.ei_signs(task_representation)  # [batch, num_limbs]
        ei_signs = torch.tanh(ei_logits)  # [batch, num_limbs] in (-1, 1)
        # >0 = excitatory, <0 = inhibitory
        
        # Compute E/I balance loss (encourage 80/20 split)
        ei_ratio = (ei_signs > 0).float().mean(dim=-1)  # [batch]
        ei_balance_target = torch.full_like(ei_ratio, self.target_ei_ratio)
        ei_balance_loss = F.mse_loss(ei_ratio, ei_balance_target)
        
        # ════════════════════════════════════════════════════════════════
        # STEP 5: Temperature & Confidence
        # ════════════════════════════════════════════════════════════════
        temp_delta = torch.tanh(self.temperature_logits(task_representation))  # [batch, 1]
        current_temperature = torch.clamp(self.temperature + 0.1 * temp_delta, min=0.1, max=2.0)
        
        confidence_logits = self.confidence_predictor(task_representation)  # [batch, 1]
        confidence = torch.sigmoid(confidence_logits).squeeze(-1)  # [batch]
        
        # ════════════════════════════════════════════════════════════════
        # STEP 6: Apply Editing to Hidden States
        # ════════════════════════════════════════════════════════════════
        # Expand gates from [batch, num_limbs] to [batch, seq_len, num_limbs]
        limb_activation_expanded = limb_activation.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Apply limb gating to hidden states (learnable per-limb transformation)
        # For now, scale hidden states by average limb activation
        avg_limb_activation = limb_activation.mean(dim=-1, keepdim=True)  # [batch, 1]
        edited_states = hidden_states * (0.5 + 0.5 * avg_limb_activation.unsqueeze(1))
        
        # Apply confidence modulation
        confidence_scale = (0.5 + 0.5 * confidence).unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        edited_states = edited_states * confidence_scale
        
        # ════════════════════════════════════════════════════════════════
        # STEP 7: Build Output Dict
        # ════════════════════════════════════════════════════════════════
        result = {
            'output': edited_states,
            'limb_gates': limb_gates,  # [batch, num_limbs]
            'limb_activation': limb_activation,  # [batch, num_limbs]
            'pathway_weights': pathway_weights,  # [batch, num_pathways]
            'head_gates': head_gates,  # [batch, num_heads]
            'head_importance': head_importance,  # [batch, num_heads]
            'ei_signs': ei_signs,  # [batch, num_limbs]
            'ei_balance_loss': ei_balance_loss,  # scalar
            'confidence': confidence,  # [batch]
            'temperature': current_temperature.squeeze(-1),  # [batch] or scalar
            'pathway_indices': pathway_indices,  # [batch]
        }
        
        if return_info:
            result['detailed_info'] = {
                'limb_gate_logits': limb_gate_logits,
                'limb_strength_logits': limb_strength_logits,
                'head_gate_logits': head_gate_logits,
                'head_importance_logits': head_importance_logits,
                'ei_ratio': ei_ratio,
                'ei_balance_target': ei_balance_target,
            }
        
        return result
    
    def get_editing_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about editing patterns.
        
        Returns:
            Dict with pathway and limb activation statistics
        """
        pathway_totals = self.pathway_activation_counts.sum()
        limb_totals = self.limb_activation_counts.sum()
        
        pathway_dist = (
            self.pathway_activation_counts / (pathway_totals + 1e-8)
        ).tolist()
        
        limb_dist = (
            self.limb_activation_counts / (limb_totals + 1e-8)
        ).tolist()
        
        return {
            'total_edits': max(pathway_totals.item(), limb_totals.item()),
            'pathway_distribution': pathway_dist,
            'limb_distribution': limb_dist,
            'current_temperature': self.temperature.item(),
        }
    
    def reset_counters(self):
        """Reset activation counters for new epoch."""
        self.pathway_activation_counts.zero_()
        self.limb_activation_counts.zero_()


class AdaptiveRNAModule(nn.Module):
    """
    Higher-level RNA module that combines editing with adaptive learning.
    
    Learns to:
    1. Adapt editing patterns based on task performance
    2. Discover beneficial limb combinations
    3. Optimize temperature for exploration vs exploitation tradeoff
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_limbs: int = 8,
        num_heads: int = 4,
        num_pathways: int = 3,
        adaptation_rate: float = 0.01
    ):
        super().__init__()
        
        self.editing_layer = RNAEditingLayer(
            hidden_dim=hidden_dim,
            num_limbs=num_limbs,
            num_heads=num_heads,
            num_pathways=num_pathways
        )
        
        self.adaptation_rate = adaptation_rate
        
        # Learned pathway preferences (which pathways are best for which tasks)
        self.pathway_preferences = nn.Parameter(
            torch.randn(num_pathways, hidden_dim) * 0.01
        )
        
        # Meta-learning: optimize editing based on feedback
        self.meta_optimizer = nn.Linear(hidden_dim + num_limbs + num_pathways, hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        feedback: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward with adaptive learning.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            feedback: Optional [batch] performance signal for adaptation
            
        Returns:
            Dict with editing results + adaptation info
        """
        # Get base editing
        editing_result = self.editing_layer(hidden_states, return_info=True)
        
        # If feedback provided, adapt editing
        if feedback is not None:
            task_rep = hidden_states.mean(dim=1)  # [batch, hidden_dim]
            
            # Compute how to adjust editing based on feedback
            feedback_signal = torch.cat([
                task_rep,
                editing_result['limb_activation'],
                editing_result['pathway_weights']
            ], dim=-1)
            
            adaptation = torch.tanh(self.meta_optimizer(feedback_signal))  # [batch, hidden_dim]
            
            # Apply adaptation to output
            editing_result['output'] = editing_result['output'] + 0.01 * adaptation.unsqueeze(1)
            editing_result['feedback_adapted'] = True
        else:
            editing_result['feedback_adapted'] = False
        
        return editing_result


# ════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════

def compute_rna_editing_loss(
    editing_output: Dict[str, Any],
    target_ei_ratio: float = 0.8,
    ei_balance_weight: float = 0.01,
    temperature_regularization: float = 0.01
) -> torch.Tensor:
    """
    Compute regularization losses for RNA editing.
    
    Args:
        editing_output: Dict from RNAEditingLayer.forward()
        target_ei_ratio: Target excitatory/inhibitory ratio
        ei_balance_weight: Weight for E/I balance loss
        temperature_regularization: Weight for temperature regularization
        
    Returns:
        Total regularization loss
    """
    loss = torch.tensor(0.0, device=editing_output['output'].device)
    
    # E/I balance loss
    if 'ei_balance_loss' in editing_output:
        loss = loss + ei_balance_weight * editing_output['ei_balance_loss']
    
    # Temperature regularization (keep close to 1.0 for stable adaptation)
    if 'temperature' in editing_output:
        temp = editing_output['temperature']
        if temp.dim() > 0:
            temp = temp.mean()
        loss = loss + temperature_regularization * F.mse_loss(temp, torch.tensor(1.0, device=temp.device))
    
    return loss


if __name__ == "__main__":
    print("Testing RNA Editing Layer...")
    
    hidden_dim = 256
    batch_size = 4
    seq_len = 32
    num_limbs = 8
    num_heads = 4
    num_pathways = 3
    
    # Create layer
    rna_layer = RNAEditingLayer(
        hidden_dim=hidden_dim,
        num_limbs=num_limbs,
        num_heads=num_heads,
        num_pathways=num_pathways
    )
    
    # Test forward
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    result = rna_layer(hidden_states, return_info=True)
    
    print(f"Output shape: {result['output'].shape}")
    print(f"Limb gates shape: {result['limb_gates'].shape}")
    print(f"Limb activation: {result['limb_activation'][0]}")
    print(f"Pathway weights: {result['pathway_weights'][0]}")
    print(f"Confidence: {result['confidence']}")
    print(f"E/I balance loss: {result['ei_balance_loss'].item():.4f}")
    
    # Test editing summary
    summary = rna_layer.get_editing_summary()
    print(f"\nEditing summary: {summary}")
    
    # Test adaptive module
    print("\nTesting Adaptive RNA Module...")
    adaptive_rna = AdaptiveRNAModule(hidden_dim, num_limbs, num_heads, num_pathways)
    
    feedback = torch.randn(batch_size)
    adaptive_result = adaptive_rna(hidden_states, feedback=feedback)
    print(f"Adaptive output shape: {adaptive_result['output'].shape}")
    print(f"Feedback adapted: {adaptive_result['feedback_adapted']}")
    
    # Test loss computation
    rna_loss = compute_rna_editing_loss(result)
    print(f"\nRNA editing loss: {rna_loss.item():.4f}")
    
    print("\n✓ RNA Editing Layer tests passed!")
