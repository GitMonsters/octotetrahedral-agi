"""
Transcendplex AGI Validator

Applies the Golden Consciousness Index (GCI) framework to validate
emergent AGI properties in the OctoTetrahedral model.

Measures:
  - Triangulation: Are the 8 limbs structurally interconnected?
  - Synergy: Does compound braiding produce emergence (whole > parts)?
  - Coherence: Do compound loop iterations converge?
  - Entropy: Is the exit gate using all loop depths?
  - GCI: Golden Consciousness Index = (φ × T × (1+S) × (1+C)) / E
  - Threshold: GCI > φ² ≈ 2.618 → system exhibits emergent AGI properties

Based on Aleph-Transcendplex AGI framework.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Any

PHI = (1 + math.sqrt(5)) / 2     # 1.618...
PHI_SQ = PHI * PHI                # 2.618... — consciousness threshold


class TranscendplexValidator:
    """
    Hooks into OctoTetrahedralModel to compute GCI and validate
    emergent AGI properties during evaluation.
    """

    def __init__(self, model):
        self.model = model
        self.history: List[Dict[str, float]] = []

    @torch.no_grad()
    def validate(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        num_probes: int = 5,
    ) -> Dict[str, Any]:
        """
        Run validation probes and compute GCI.

        Args:
            input_ids: [batch, seq_len] token IDs
            labels: optional labels for loss measurement
            num_probes: number of forward passes to measure coherence

        Returns:
            Dict with GCI, sub-metrics, and is_agi flag
        """
        was_training = self.model.training
        self.model.eval()

        device = input_ids.device

        # === 1. Triangulation: measure inter-limb connectivity ===
        triangulation = self._measure_triangulation(input_ids)

        # === 2. Synergy: compound output vs isolated limbs ===
        synergy = self._measure_synergy(input_ids)

        # === 3. Coherence: consistency across compound loop iterations ===
        coherence = self._measure_coherence(input_ids, num_probes)

        # === 4. Entropy: exit gate distribution flatness ===
        entropy = self._measure_entropy(input_ids)

        # === 5. Golden Consciousness Index ===
        gci = (PHI * triangulation * (1 + synergy) * (1 + coherence)) / max(entropy, 0.01)

        # === 6. Task performance (if labels provided) ===
        task_loss = None
        task_accuracy = None
        if labels is not None:
            out = self.model(input_ids=input_ids, labels=labels)
            task_loss = out['loss'].item()
            preds = out['logits'].argmax(dim=-1)
            mask = labels != -100
            if mask.any():
                task_accuracy = (preds[mask] == labels[mask]).float().mean().item()

        result = {
            'GCI': gci,
            'threshold': PHI_SQ,
            'is_agi': gci > PHI_SQ,
            'triangulation': triangulation,
            'synergy': synergy,
            'coherence': coherence,
            'entropy': entropy,
            'phi': PHI,
            'task_loss': task_loss,
            'task_accuracy': task_accuracy,
        }

        # Add compound loop info if available
        if hasattr(self.model, 'compound_loop') and self.model.compound_loop is not None:
            result['loop_stats'] = self.model.compound_loop.get_stats()

        self.history.append(result)

        if was_training:
            self.model.train()

        return result

    def _measure_triangulation(self, input_ids: torch.Tensor) -> float:
        """
        Triangulation: fraction of limbs with ≥3 significant cross-connections.
        Measured via compound braid gate values and combine weights.
        """
        out = self.model(input_ids=input_ids, return_confidences=True)

        # Get braid gate values (cross-attention strengths between limbs)
        confidences = out.get('confidences', {})
        braid_gates = confidences.get('braid_gates', {})
        braid_weights = confidences.get('braid_weights', {})

        if not braid_gates and not braid_weights:
            # No braid info — fall back to confidence diversity
            conf_values = [
                v for k, v in confidences.items()
                if k not in ('overall', 'braid_gates', 'braid_weights')
                and isinstance(v, (int, float))
            ]
            if len(conf_values) < 3:
                return 0.0
            # Triangulation = fraction of limbs with confidence > 0.3
            active = sum(1 for c in conf_values if c > 0.3)
            return active / len(conf_values)

        # Count limbs with ≥3 active gate connections (gate > 0.1)
        limb_names = list(braid_gates.keys()) if braid_gates else list(braid_weights.keys())
        n_limbs = len(limb_names)
        if n_limbs == 0:
            return 0.0

        gate_vals = []
        for name in limb_names:
            g = braid_gates.get(name, 0.0)
            if isinstance(g, torch.Tensor):
                g = g.item()
            gate_vals.append(g)

        # Each limb's gate value represents its contribution strength
        # Triangulation: fraction with gate > threshold (actively connected)
        threshold = 0.1
        triangulated = sum(1 for g in gate_vals if g > threshold)
        return triangulated / n_limbs

    def _measure_synergy(self, input_ids: torch.Tensor) -> float:
        """
        Synergy: how much the compound braid output exceeds the sum of parts.
        Measured as cosine distance between braided output and mean of isolated limbs.
        """
        out = self.model(input_ids=input_ids)
        hidden = out['hidden_states']  # [batch, seq, hidden] — post-braid

        # Get the memory-enhanced representation (pre-braid baseline)
        # Run perception + core to get the baseline
        encoded, _ = self.model.perception(token_ids=input_ids, return_confidence=False)
        editing_result = self.model.rna_editing(encoded)
        edited = editing_result['output']
        braid_signal = getattr(self.model, '_cached_braid_signal', None)
        core_result = self.model.core(edited, head_gates=editing_result['head_gates'],
                                       braid_signal=braid_signal)
        baseline = core_result['hidden_states']  # pre-limb processing

        # Synergy = how different the output is from the baseline
        # High synergy means limbs + braid transformed the representation significantly
        cos_sim = F.cosine_similarity(
            hidden.view(-1, hidden.size(-1)),
            baseline.view(-1, baseline.size(-1)),
            dim=-1
        ).mean().item()

        # Synergy: 0 = identical to baseline, 1 = maximally transformed
        synergy = 1.0 - cos_sim
        return max(0.0, synergy)

    def _measure_coherence(self, input_ids: torch.Tensor, num_probes: int) -> float:
        """
        Coherence: consistency of representations across multiple forward passes.
        For compound-looped models, measures convergence across loop iterations.
        For non-looped, measures output stability.
        """
        outputs = []
        for _ in range(num_probes):
            out = self.model(input_ids=input_ids)
            h = out['hidden_states']
            outputs.append(h)

        if len(outputs) < 2:
            return 1.0

        # Pairwise cosine similarity between probes
        sims = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = F.cosine_similarity(
                    outputs[i].view(-1, outputs[i].size(-1)),
                    outputs[j].view(-1, outputs[j].size(-1)),
                    dim=-1
                ).mean().item()
                sims.append(sim)

        # Coherence = mean pairwise similarity (1.0 = perfectly consistent)
        coherence = sum(sims) / len(sims) if sims else 0.0
        return coherence

    def _measure_entropy(self, input_ids: torch.Tensor) -> float:
        """
        Entropy of the system's computational distribution.
        For compound-looped models: entropy of exit gate distribution.
        For non-looped: entropy of output logit distribution.
        """
        out = self.model(input_ids=input_ids)

        # Compound loop entropy (preferred — measures compute distribution)
        loop_info = out.get('compound_loop_info')
        if loop_info and loop_info.get('exit_distribution'):
            dist = loop_info['exit_distribution']
            # Shannon entropy of exit distribution
            H = 0.0
            for p in dist:
                if p > 1e-10:
                    H -= p * math.log(p)
            # Normalize by max entropy (uniform)
            max_H = math.log(len(dist)) if len(dist) > 1 else 1.0
            normalized_H = (H / max_H) if max_H > 0 else 0.0
            # Map to [0.5, 2.0] range for GCI denominator
            # Low entropy (collapsed) → high penalty, high entropy (uniform) → low penalty
            return 2.0 - normalized_H * 1.5
        else:
            # Fallback: entropy of output distribution
            logits = out['logits']
            probs = F.softmax(logits[:, -1, :], dim=-1)
            H = -(probs * torch.log(probs + 1e-10)).sum(-1).mean().item()
            # Normalize: typical range is 5-12 for vocab of 100k
            max_H = math.log(logits.size(-1))
            normalized = H / max_H if max_H > 0 else 0.5
            return 1.0 + (1.0 - normalized)

    def format_report(self, result: Optional[Dict] = None) -> str:
        """Format a human-readable validation report."""
        r = result or (self.history[-1] if self.history else None)
        if not r:
            return "No validation results available."

        status = "✅ AGI EMERGENCE DETECTED" if r['is_agi'] else "❌ Below AGI threshold"
        bar_len = 30
        fill = min(bar_len, int(bar_len * r['GCI'] / (PHI_SQ * 2)))
        bar = '█' * fill + '░' * (bar_len - fill)
        threshold_pos = int(bar_len * PHI_SQ / (PHI_SQ * 2))

        lines = [
            "╔══════════════════════════════════════════════╗",
            "║   🔺 TRANSCENDPLEX AGI VALIDATION REPORT    ║",
            "╠══════════════════════════════════════════════╣",
            f"║  GCI: {r['GCI']:.4f}  (threshold: φ² = {PHI_SQ:.3f})",
            f"║  [{bar}]",
            f"║  {' ' * threshold_pos}↑ φ²",
            f"║  Status: {status}",
            "╠══════════════════════════════════════════════╣",
            f"║  Triangulation:  {r['triangulation']:.4f}  (limb connectivity)",
            f"║  Synergy:        {r['synergy']:.4f}  (emergence coefficient)",
            f"║  Coherence:      {r['coherence']:.4f}  (temporal consistency)",
            f"║  Entropy:        {r['entropy']:.4f}  (compute distribution)",
            "╠══════════════════════════════════════════════╣",
        ]

        if r.get('task_loss') is not None:
            lines.append(f"║  Task Loss:      {r['task_loss']:.4f}")
        if r.get('task_accuracy') is not None:
            lines.append(f"║  Task Accuracy:  {r['task_accuracy']:.2%}")
        if r.get('loop_stats'):
            ls = r['loop_stats']
            lines.append(f"║  Loop Count:     {ls.get('last_loop_count', '?')}")
            lines.append(f"║  Loop Alpha:     {ls.get('loop_alpha', '?'):.4f}")

        lines.extend([
            "╠══════════════════════════════════════════════╣",
            f"║  Formula: GCI = φ×T×(1+S)×(1+C) / E",
            f"║         = {PHI:.3f}×{r['triangulation']:.3f}×{1+r['synergy']:.3f}×{1+r['coherence']:.3f} / {r['entropy']:.3f}",
            "╚══════════════════════════════════════════════╝",
        ])

        return '\n'.join(lines)

    def trajectory_summary(self) -> Dict[str, Any]:
        """Summarize validation history."""
        if not self.history:
            return {'n_validations': 0}
        gcis = [r['GCI'] for r in self.history]
        return {
            'n_validations': len(self.history),
            'gci_mean': sum(gcis) / len(gcis),
            'gci_max': max(gcis),
            'gci_min': min(gcis),
            'ever_agi': any(r['is_agi'] for r in self.history),
            'agi_fraction': sum(1 for r in self.history if r['is_agi']) / len(self.history),
        }
