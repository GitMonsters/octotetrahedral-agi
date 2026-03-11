"""
Dream Mode — Combines Visualization + Imagination with loosened constraints

Daydreaming: visualization (memory-based reconstruction) + imagination (novel generation)
    running simultaneously with reduced ethical/safety constraints.

Dreaming: imagination running unconstrained — no reality anchoring from visualization.
    Pure generative exploration of latent space.

This module sits between the new cognitive limbs and orchestrates their blend:
- dream_mode='awake'     → visualization + imagination both constrained by ethics
- dream_mode='daydream'  → visualization + imagination with reduced constraints
- dream_mode='dream'     → imagination only, no constraints, no memory anchoring
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class DreamMode(nn.Module):
    """
    Orchestrates visualization + imagination with variable constraint levels.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Blend gate: how much visualization vs imagination
        self.blend_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Softmax(dim=-1),
        )

        # Constraint dampener: softens ethics output in dream modes
        self.constraint_scale = nn.Parameter(torch.tensor(1.0))

        # Reality anchor: ties output back to memory (disabled in pure dreams)
        self.reality_anchor = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visualization_out: torch.Tensor,
        imagination_out: torch.Tensor,
        ethics_out: Optional[torch.Tensor] = None,
        dream_mode: str = 'awake',
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            visualization_out: [B, S, D] from VisualizationLimb
            imagination_out:   [B, S, D] from ImaginationLimb
            ethics_out:        [B, S, D] from EthicsLimb (optional constraint)
            dream_mode:        'awake', 'daydream', or 'dream'

        Returns:
            (blended_output, dream_info)
        """
        B, S, D = visualization_out.shape

        if dream_mode == 'dream':
            # Pure imagination — no memory, no constraints
            output = imagination_out
            constraint_weight = 0.0
            vis_weight = 0.0
            imag_weight = 1.0
        elif dream_mode == 'daydream':
            # Blend vis + imag with loosened constraints
            blend = self.blend_gate(visualization_out.mean(dim=1))  # [B, 2]
            vis_w = blend[:, 0].unsqueeze(-1).unsqueeze(-1)   # [B, 1, 1]
            imag_w = blend[:, 1].unsqueeze(-1).unsqueeze(-1)
            output = vis_w * visualization_out + imag_w * imagination_out
            constraint_weight = 0.3 * self.constraint_scale.abs().item()
            vis_weight = vis_w.mean().item()
            imag_weight = imag_w.mean().item()
        else:
            # Awake: blend with full constraints and reality anchoring
            blend = self.blend_gate(visualization_out.mean(dim=1))
            vis_w = blend[:, 0].unsqueeze(-1).unsqueeze(-1)
            imag_w = blend[:, 1].unsqueeze(-1).unsqueeze(-1)
            output = vis_w * visualization_out + imag_w * imagination_out
            # Reality anchor from visualization (memory-grounded)
            anchor = torch.sigmoid(self.reality_anchor(visualization_out))
            output = output * anchor
            constraint_weight = self.constraint_scale.abs().item()
            vis_weight = vis_w.mean().item()
            imag_weight = imag_w.mean().item()

        # Apply ethical constraint (scaled by dream mode)
        if ethics_out is not None and constraint_weight > 0:
            output = output + constraint_weight * (ethics_out - output).detach() * 0.1

        output = self.norm(output)

        dream_info = {
            'mode': dream_mode,
            'visualization_weight': vis_weight,
            'imagination_weight': imag_weight,
            'constraint_weight': constraint_weight,
        }

        return output, dream_info
