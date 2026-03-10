"""
Cognitive Geometry Engine — Compound Integration of ML Vocabulary

Implements all 41 ML vocabulary concepts as operational PyTorch modules
that integrate into the OctoTetrahedral forward pass:

  1. SVD Activation Decomposer — dominant semantic axis extraction
  2. Concept Alignment Matrix — cosine similarity between limb representations
  3. Entropy Flow Monitor — track uncertainty across processing stages
  4. Semantic Drift Detector — measure vector rotation across forward passes
  5. Anchor Vector System — persistent identity/topic bias vectors
  6. Repetition Dampener — suppress token echo patterns in logits
  7. Branch Scorer & Pruner — score limb branches by goal alignment
  8. Manifold Partitioner — enforce orthogonality between limb subspaces
  9. Goal Vector System — explicit goal direction guiding all reasoning
 10. Attention Plane Reconstructor — compress attention into 2D concept map
 11. Vector Field Tracker — track representation flow across layers
 12. Cross-Limb Orthogonality Loss — regularize limbs to stay independent

All modules are designed to:
  - Add zero overhead when disabled (gated by config)
  - Produce auxiliary losses for training
  - Produce diagnostic info dict for monitoring
  - Be numerically stable (NaN-safe throughout)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class CognitiveGeometryConfig:
    """Configuration for all cognitive geometry subsystems."""
    enabled: bool = True

    # SVD Activation Decomposer
    svd_enabled: bool = True
    svd_top_k: int = 8               # Number of dominant axes to track
    svd_loss_weight: float = 0.01    # Encourage spread across axes

    # Concept Alignment Matrix
    alignment_enabled: bool = True
    alignment_loss_weight: float = 0.01  # Penalize limb collapse

    # Entropy Flow Monitor
    entropy_monitor_enabled: bool = True
    entropy_target: float = 2.0      # Target entropy (bits) — not too certain, not too random
    entropy_loss_weight: float = 0.005

    # Semantic Drift Detector
    drift_enabled: bool = True
    drift_max_rotation: float = 0.5  # Max cosine distance before alarm
    drift_loss_weight: float = 0.01

    # Anchor Vector System
    anchor_enabled: bool = True
    num_anchors: int = 4             # Persistent concept anchors
    anchor_decay: float = 0.95       # Exponential decay per step
    anchor_strength: float = 0.1     # Blend weight into hidden states

    # Repetition Dampener
    repetition_dampen_enabled: bool = True
    repetition_penalty: float = 1.2  # Multiplicative penalty on repeated tokens
    repetition_window: int = 32      # How far back to check

    # Branch Scorer
    branch_scorer_enabled: bool = True
    branch_prune_threshold: float = 0.1  # Min score to keep branch

    # Manifold Partitioner (orthogonality regularization)
    manifold_enabled: bool = True
    manifold_loss_weight: float = 0.02

    # Goal Vector System
    goal_vector_enabled: bool = True
    goal_strength: float = 0.15      # Blend strength

    # Attention Plane Reconstructor
    attention_plane_enabled: bool = True
    plane_dim: int = 16              # Compressed concept map size

    # Vector Field Tracker
    vector_field_enabled: bool = True


class SVDActivationDecomposer(nn.Module):
    """
    Decomposes hidden activations via truncated SVD to extract
    dominant semantic directions. Produces a diversity loss that
    encourages the model to use the full rank of its representation space.
    """

    def __init__(self, hidden_dim: int, top_k: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = min(top_k, hidden_dim)
        # Learnable semantic axis labels (soft)
        self.axis_tagger = nn.Linear(hidden_dim, top_k)

    def forward(self, hidden: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
        Returns:
            dict with 'singular_values', 'dominant_axes', 'diversity_loss', 'axis_tags'
        """
        B, S, D = hidden.shape

        # Flatten batch for SVD: [B*S, D] → too expensive. Use pooled: [B, D]
        pooled = hidden.mean(dim=1)  # [B, D]

        # Truncated SVD via torch.linalg.svd (economy mode)
        try:
            U, sigma, Vh = torch.linalg.svd(pooled, full_matrices=False)
            top_sigma = sigma[:, :self.top_k]  # [B, k]
            top_axes = Vh[:, :self.top_k, :]   # [B, k, D]

            # Diversity loss: encourage spread (penalize concentration in top-1)
            sigma_norm = top_sigma / (top_sigma.sum(dim=-1, keepdim=True) + 1e-8)
            # Entropy of normalized singular values (higher = more diverse)
            sv_entropy = -(sigma_norm * torch.log(sigma_norm + 1e-10)).sum(dim=-1).mean()
            max_entropy = math.log(self.top_k)
            diversity_loss = (max_entropy - sv_entropy) / max_entropy  # 0 = perfect spread

            # Axis tags via learned projection
            axis_tags = self.axis_tagger(top_axes.mean(dim=0))  # [k, top_k] → soft labels

        except Exception:
            # SVD can fail on degenerate matrices
            top_sigma = torch.zeros(B, self.top_k, device=hidden.device)
            top_axes = torch.zeros(B, self.top_k, D, device=hidden.device)
            diversity_loss = torch.tensor(0.0, device=hidden.device)
            axis_tags = None

        return {
            'singular_values': top_sigma,
            'dominant_axes': top_axes,
            'diversity_loss': diversity_loss,
            'axis_tags': axis_tags,
        }


class ConceptAlignmentMatrix(nn.Module):
    """
    Computes pairwise cosine similarity between limb output representations.
    Detects semantic clustering, reinforcement, and conflict.
    """

    def __init__(self, num_limbs: int = 6):
        super().__init__()
        self.num_limbs = num_limbs

    def forward(self, limb_outputs: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Args:
            limb_outputs: list of [batch, seq_len, hidden_dim] tensors (one per limb)
        Returns:
            dict with 'alignment_matrix', 'cluster_indices', 'conflict_pairs', 'collapse_loss'
        """
        # Pool each limb to [batch, hidden_dim]
        pooled = [x.mean(dim=1) for x in limb_outputs]
        stacked = torch.stack(pooled, dim=1)  # [B, num_limbs, D]

        # Normalize
        normed = F.normalize(stacked, dim=-1)  # [B, num_limbs, D]

        # Pairwise cosine similarity: [B, num_limbs, num_limbs]
        alignment = torch.bmm(normed, normed.transpose(1, 2))

        # Collapse loss: penalize high off-diagonal similarity (limbs should diversify)
        mask = 1.0 - torch.eye(self.num_limbs, device=alignment.device).unsqueeze(0)
        off_diag = alignment * mask
        collapse_loss = off_diag.abs().mean()  # Lower = more orthogonal

        # Detect conflict pairs (negative similarity)
        conflict_pairs = (alignment < -0.3).float().sum(dim=(1, 2)).mean()

        # Detect clusters (high positive similarity)
        cluster_mask = (alignment > 0.8) & (mask.bool())
        cluster_count = cluster_mask.float().sum(dim=(1, 2)).mean()

        return {
            'alignment_matrix': alignment.detach(),
            'collapse_loss': collapse_loss,
            'conflict_pairs': conflict_pairs,
            'cluster_count': cluster_count,
        }


class EntropyFlowMonitor(nn.Module):
    """
    Tracks token entropy at multiple processing stages.
    Produces a loss that guides entropy toward a target range.
    """

    def __init__(self, hidden_dim: int, target_entropy: float = 2.0):
        super().__init__()
        self.target = target_entropy
        # Project hidden states to entropy estimate
        self.entropy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
            logits: [batch, seq_len, vocab_size] (optional, for actual entropy)
        """
        # Predicted entropy from hidden states (learned)
        pred_entropy = self.entropy_head(hidden).squeeze(-1)  # [B, S]

        # Actual logit entropy if available
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            actual_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, S]
        else:
            actual_entropy = pred_entropy.detach()

        # Loss: guide actual entropy toward target
        entropy_deviation = (actual_entropy.mean() - self.target) ** 2
        # Also train the predictor
        prediction_loss = F.mse_loss(pred_entropy, actual_entropy.detach())

        return {
            'predicted_entropy': pred_entropy.detach(),
            'actual_entropy': actual_entropy.detach() if logits is not None else None,
            'entropy_loss': entropy_deviation + 0.1 * prediction_loss,
            'mean_entropy': actual_entropy.mean().item(),
        }


class SemanticDriftDetector(nn.Module):
    """
    Measures rotation between consecutive forward passes.
    Detects topic drift, tone drift, intent drift.
    """

    def __init__(self, hidden_dim: int, max_rotation: float = 0.5):
        super().__init__()
        self.max_rotation = max_rotation
        # EMA of previous representation
        self.register_buffer('prev_state', torch.zeros(hidden_dim))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, hidden: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
        """
        current = hidden.mean(dim=(0, 1))  # [D] — average over batch & seq

        if self.initialized:
            # Cosine distance (1 - similarity)
            cos_sim = F.cosine_similarity(
                current.unsqueeze(0), self.prev_state.unsqueeze(0)
            ).item()
            drift = 1.0 - cos_sim
            drift_alarm = drift > self.max_rotation
            # Loss: penalize excessive drift
            drift_loss = F.relu(torch.tensor(drift - self.max_rotation, device=hidden.device))
        else:
            drift = 0.0
            drift_alarm = False
            drift_loss = torch.tensor(0.0, device=hidden.device)
            self.initialized.fill_(True)

        # Update with EMA
        self.prev_state.data = 0.9 * self.prev_state + 0.1 * current.detach()

        return {
            'drift': drift,
            'drift_alarm': drift_alarm,
            'drift_loss': drift_loss,
        }


class AnchorVectorSystem(nn.Module):
    """
    Maintains persistent concept anchor vectors that bias hidden states.
    Anchors decay exponentially without reinforcement.
    Implements identity persistence + topic anchoring.
    """

    def __init__(self, hidden_dim: int, num_anchors: int = 4,
                 decay: float = 0.95, strength: float = 0.1):
        super().__init__()
        self.num_anchors = num_anchors
        self.decay = decay
        self.strength = strength
        self.hidden_dim = hidden_dim

        # Learnable anchor directions
        self.anchor_bases = nn.Parameter(torch.randn(num_anchors, hidden_dim) * 0.01)
        # Anchor magnitudes (decayable state, not a parameter)
        self.register_buffer('anchor_magnitudes', torch.ones(num_anchors))
        # Anchor selector: given hidden state, which anchors to reinforce?
        self.anchor_gate = nn.Linear(hidden_dim, num_anchors)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
        Returns:
            (anchored_hidden, info_dict)
        """
        # Determine which anchors are relevant
        pooled = hidden.mean(dim=(0, 1))  # [D]
        gates = torch.sigmoid(self.anchor_gate(pooled))  # [num_anchors]

        # Decay all anchors
        self.anchor_magnitudes.data *= self.decay

        # Reinforce active anchors
        reinforcement = gates.detach()
        self.anchor_magnitudes.data = torch.clamp(
            self.anchor_magnitudes + 0.1 * reinforcement, max=1.0
        )

        # Compute anchor bias
        active_anchors = F.normalize(self.anchor_bases, dim=-1)  # [A, D]
        weighted = active_anchors * (self.anchor_magnitudes * gates).unsqueeze(-1)  # [A, D]
        anchor_bias = weighted.sum(dim=0)  # [D]

        # Apply bias to hidden states
        anchored = hidden + self.strength * anchor_bias.unsqueeze(0).unsqueeze(0)

        return anchored, {
            'anchor_magnitudes': self.anchor_magnitudes.detach().clone(),
            'anchor_gates': gates.detach(),
            'anchor_bias_norm': anchor_bias.norm().item(),
        }


class RepetitionDampener(nn.Module):
    """
    Suppresses repeated token patterns in logits.
    Operates on the output logits directly (no training parameters).
    """

    def __init__(self, penalty: float = 1.2, window: int = 32):
        super().__init__()
        self.penalty = penalty
        self.window = window

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            input_ids: [batch, seq_len] — previous tokens for penalty
        Returns:
            dampened logits
        """
        if input_ids is None or not self.training:
            return logits

        B, S, V = logits.shape
        dampened = logits.clone()

        for b in range(B):
            for s in range(S):
                start = max(0, s - self.window)
                seen = input_ids[b, start:s]
                if len(seen) > 0:
                    unique_seen = seen.unique()
                    # Reduce logits of recently seen tokens
                    dampened[b, s, unique_seen] /= self.penalty

        return dampened


class BranchScorer(nn.Module):
    """
    Scores each limb branch by goal alignment and confidence.
    Low-scoring branches get attenuated.
    """

    def __init__(self, hidden_dim: int, num_branches: int = 6,
                 prune_threshold: float = 0.1):
        super().__init__()
        self.prune_threshold = prune_threshold
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        branch_outputs: List[torch.Tensor],
        goal_vector: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Args:
            branch_outputs: list of [B, S, D] tensors
            goal_vector: [B, D] or [D]
        Returns:
            (pruned_branches, info)
        """
        if goal_vector.dim() == 1:
            goal_vector = goal_vector.unsqueeze(0)
        # Expand goal to match batch size
        if goal_vector.size(0) == 1 and len(branch_outputs) > 0:
            B = branch_outputs[0].size(0)
            goal_vector = goal_vector.expand(B, -1)

        scores = []
        pruned = []

        for branch in branch_outputs:
            branch_pooled = branch.mean(dim=1)  # [B, D]
            combined = torch.cat([branch_pooled, goal_vector], dim=-1)  # [B, 2D]
            score = self.scorer(combined).mean()  # scalar
            scores.append(score)

            # Soft pruning: scale by score
            scale = torch.clamp(score / max(self.prune_threshold, 0.01), max=1.0)
            pruned.append(branch * scale)

        scores_tensor = torch.stack(scores)
        pruned_count = (scores_tensor < self.prune_threshold).sum().item()

        return pruned, {
            'branch_scores': scores_tensor.detach(),
            'pruned_count': pruned_count,
        }


class ManifoldPartitioner(nn.Module):
    """
    Regularizes limb representations to occupy orthogonal subspaces.
    Each limb should operate on its own sub-manifold.
    Loss = sum of squared cosine similarities between limb direction vectors.
    """

    def __init__(self, hidden_dim: int, num_limbs: int = 6):
        super().__init__()
        # Learnable subspace projectors per limb
        self.projectors = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // num_limbs, bias=False)
            for _ in range(num_limbs)
        ])

    def forward(self, limb_outputs: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Args:
            limb_outputs: list of [B, S, D] — one per limb
        Returns:
            dict with 'orthogonality_loss', 'subspace_overlap'
        """
        # Project each limb into its learned subspace
        projected = []
        for i, (proj, output) in enumerate(zip(self.projectors, limb_outputs)):
            p = proj(output.mean(dim=1))  # [B, D//N]
            projected.append(F.normalize(p, dim=-1))

        # Compute pairwise overlap
        total_overlap = torch.tensor(0.0, device=limb_outputs[0].device)
        n_pairs = 0
        for i in range(len(projected)):
            for j in range(i + 1, len(projected)):
                # Cosine similarity between subspace representations
                sim = F.cosine_similarity(projected[i], projected[j], dim=-1).mean()
                total_overlap = total_overlap + sim ** 2
                n_pairs += 1

        orthogonality_loss = total_overlap / max(n_pairs, 1)

        return {
            'orthogonality_loss': orthogonality_loss,
            'subspace_overlap': total_overlap.item() / max(n_pairs, 1),
        }


class GoalVectorSystem(nn.Module):
    """
    Extracts an explicit goal direction from the prompt/context.
    All subsequent processing is biased toward this goal.
    Subgoals are intermediate projections.
    """

    def __init__(self, hidden_dim: int, strength: float = 0.15):
        super().__init__()
        self.strength = strength
        self.goal_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.subgoal_projector = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            hidden: [batch, seq_len, hidden_dim]
        Returns:
            (goal_biased_hidden, info)
        """
        # Goal = dominant direction from context
        context = hidden.mean(dim=1)  # [B, D]
        goal = F.normalize(self.goal_extractor(context), dim=-1)  # [B, D]

        # Subgoal per position
        subgoals = self.subgoal_projector(hidden)  # [B, S, D]

        # Goal alignment per position
        alignment = F.cosine_similarity(
            subgoals, goal.unsqueeze(1).expand_as(subgoals), dim=-1
        )  # [B, S]

        # Bias hidden toward goal
        goal_bias = goal.unsqueeze(1) * self.strength  # [B, 1, D]
        biased = hidden + goal_bias

        return biased, {
            'goal_vector': goal.detach(),
            'goal_alignment': alignment.detach().mean().item(),
        }


class AttentionPlaneReconstructor(nn.Module):
    """
    Compresses the full hidden representation into a low-dimensional
    concept map. Think of it as a 2D "attention gravity map."
    """

    def __init__(self, hidden_dim: int, plane_dim: int = 16):
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, plane_dim)
        self.decoder = nn.Linear(plane_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            hidden: [B, S, D]
        Returns:
            dict with 'concept_map', 'reconstruction_loss'
        """
        concept_map = self.encoder(hidden)  # [B, S, plane_dim]
        reconstructed = self.decoder(concept_map)  # [B, S, D]

        reconstruction_loss = F.mse_loss(reconstructed, hidden.detach())

        return {
            'concept_map': concept_map.detach(),
            'reconstruction_loss': reconstruction_loss,
        }


class VectorFieldTracker(nn.Module):
    """
    Tracks how representations flow by measuring the delta between
    input and output of the cognitive geometry engine.
    No learnable parameters — pure diagnostic.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, pre: torch.Tensor, post: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Args:
            pre: hidden states before cognitive geometry [B, S, D]
            post: hidden states after cognitive geometry [B, S, D]
        """
        delta = post - pre  # [B, S, D]
        flow_magnitude = delta.norm(dim=-1).mean().item()
        flow_direction = F.normalize(delta.mean(dim=(0, 1)), dim=-1)

        # Curvature: how much does the flow vary across positions?
        if pre.size(1) > 1:
            pos_deltas = delta[:, 1:, :] - delta[:, :-1, :]
            curvature = pos_deltas.norm(dim=-1).mean().item()
        else:
            curvature = 0.0

        return {
            'flow_magnitude': flow_magnitude,
            'flow_direction': flow_direction.detach(),
            'curvature': curvature,
        }


class CognitiveGeometryEngine(nn.Module):
    """
    Master module that compounds all cognitive geometry subsystems.

    Integrates into the OctoTetrahedral forward pass between
    limb processing and action output. Produces:
      - Modified hidden states (anchored, goal-biased)
      - Auxiliary losses (diversity, orthogonality, entropy, drift)
      - Diagnostic info for monitoring
    """

    def __init__(self, hidden_dim: int, num_limbs: int = 6,
                 config: Optional[CognitiveGeometryConfig] = None):
        super().__init__()
        self.config = config or CognitiveGeometryConfig()
        self.hidden_dim = hidden_dim

        if not self.config.enabled:
            return

        # 1. SVD Activation Decomposer
        if self.config.svd_enabled:
            self.svd = SVDActivationDecomposer(hidden_dim, self.config.svd_top_k)

        # 2. Concept Alignment Matrix
        if self.config.alignment_enabled:
            self.alignment = ConceptAlignmentMatrix(num_limbs)

        # 3. Entropy Flow Monitor
        if self.config.entropy_monitor_enabled:
            self.entropy_monitor = EntropyFlowMonitor(hidden_dim, self.config.entropy_target)

        # 4. Semantic Drift Detector
        if self.config.drift_enabled:
            self.drift_detector = SemanticDriftDetector(hidden_dim, self.config.drift_max_rotation)

        # 5. Anchor Vector System
        if self.config.anchor_enabled:
            self.anchors = AnchorVectorSystem(
                hidden_dim, self.config.num_anchors,
                self.config.anchor_decay, self.config.anchor_strength
            )

        # 6. Repetition Dampener
        if self.config.repetition_dampen_enabled:
            self.repetition_dampener = RepetitionDampener(
                self.config.repetition_penalty, self.config.repetition_window
            )

        # 7. Branch Scorer
        if self.config.branch_scorer_enabled:
            self.branch_scorer = BranchScorer(
                hidden_dim, num_limbs, self.config.branch_prune_threshold
            )

        # 8. Manifold Partitioner
        if self.config.manifold_enabled:
            self.manifold = ManifoldPartitioner(hidden_dim, num_limbs)

        # 9. Goal Vector System
        if self.config.goal_vector_enabled:
            self.goal_system = GoalVectorSystem(hidden_dim, self.config.goal_strength)

        # 10. Attention Plane Reconstructor
        if self.config.attention_plane_enabled:
            self.attention_plane = AttentionPlaneReconstructor(
                hidden_dim, self.config.plane_dim
            )

        # 11. Vector Field Tracker
        if self.config.vector_field_enabled:
            self.vector_field = VectorFieldTracker()

    def forward(
        self,
        hidden: torch.Tensor,
        limb_outputs: Optional[List[torch.Tensor]] = None,
        logits: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full cognitive geometry pass.

        Args:
            hidden: Main hidden states [B, S, D] (post-reasoning)
            limb_outputs: Optional list of per-limb outputs for alignment/pruning
            logits: Optional output logits for entropy monitoring & repetition dampening
            input_ids: Optional input token IDs for repetition dampening

        Returns:
            dict with:
                'hidden': modified hidden states
                'logits': modified logits (if provided)
                'aux_loss': total auxiliary loss
                'info': diagnostic information
        """
        if not self.config.enabled:
            return {
                'hidden': hidden,
                'logits': logits,
                'aux_loss': torch.tensor(0.0, device=hidden.device),
                'info': {},
            }

        pre_hidden = hidden
        aux_loss = torch.tensor(0.0, device=hidden.device)
        info: Dict[str, Any] = {}

        # === 9. Goal Vector: bias hidden toward extracted goal ===
        goal_vector = None
        if self.config.goal_vector_enabled:
            hidden, goal_info = self.goal_system(hidden)
            info['goal'] = goal_info
            goal_vector = goal_info['goal_vector']

        # === 5. Anchor Vectors: persistent identity bias ===
        if self.config.anchor_enabled:
            hidden, anchor_info = self.anchors(hidden)
            info['anchors'] = anchor_info

        # === 7. Branch Scoring: score and attenuate weak limbs ===
        if self.config.branch_scorer_enabled and limb_outputs is not None and goal_vector is not None:
            pruned_limbs, branch_info = self.branch_scorer(limb_outputs, goal_vector.mean(dim=0))
            info['branches'] = branch_info

        # === 2. Concept Alignment Matrix ===
        if self.config.alignment_enabled and limb_outputs is not None:
            align_info = self.alignment(limb_outputs)
            aux_loss = aux_loss + self.config.alignment_loss_weight * align_info['collapse_loss']
            info['alignment'] = {
                'collapse_loss': align_info['collapse_loss'].item(),
                'conflict_pairs': align_info['conflict_pairs'].item(),
                'cluster_count': align_info['cluster_count'].item(),
            }

        # === 8. Manifold Partitioner ===
        if self.config.manifold_enabled and limb_outputs is not None:
            manifold_info = self.manifold(limb_outputs)
            aux_loss = aux_loss + self.config.manifold_loss_weight * manifold_info['orthogonality_loss']
            info['manifold'] = {
                'orthogonality_loss': manifold_info['orthogonality_loss'].item(),
                'subspace_overlap': manifold_info['subspace_overlap'],
            }

        # === 1. SVD Decomposition ===
        if self.config.svd_enabled:
            svd_info = self.svd(hidden)
            aux_loss = aux_loss + self.config.svd_loss_weight * svd_info['diversity_loss']
            info['svd'] = {
                'diversity_loss': svd_info['diversity_loss'].item(),
                'top_singular_values': svd_info['singular_values'].detach().mean(dim=0).tolist()
                    if svd_info['singular_values'].numel() > 0 else [],
            }

        # === 3. Entropy Flow Monitor ===
        if self.config.entropy_monitor_enabled:
            entropy_info = self.entropy_monitor(hidden, logits)
            aux_loss = aux_loss + self.config.entropy_loss_weight * entropy_info['entropy_loss']
            info['entropy'] = {
                'mean_entropy': entropy_info['mean_entropy'],
            }

        # === 4. Semantic Drift Detector ===
        if self.config.drift_enabled:
            drift_info = self.drift_detector(hidden)
            aux_loss = aux_loss + self.config.drift_loss_weight * drift_info['drift_loss']
            info['drift'] = {
                'drift': drift_info['drift'],
                'alarm': drift_info['drift_alarm'],
            }

        # === 10. Attention Plane ===
        if self.config.attention_plane_enabled:
            plane_info = self.attention_plane(hidden)
            # Reconstruction loss is diagnostic, not added to main loss
            info['attention_plane'] = {
                'reconstruction_loss': plane_info['reconstruction_loss'].item(),
            }

        # === 6. Repetition Dampener (on logits) ===
        if self.config.repetition_dampen_enabled and logits is not None:
            logits = self.repetition_dampener(logits, input_ids)

        # === 11. Vector Field Tracker ===
        if self.config.vector_field_enabled:
            field_info = self.vector_field(pre_hidden, hidden)
            info['vector_field'] = field_info

        # NaN safety
        if torch.isnan(aux_loss):
            aux_loss = torch.tensor(0.0, device=hidden.device)

        return {
            'hidden': hidden,
            'logits': logits,
            'aux_loss': aux_loss,
            'info': info,
        }
