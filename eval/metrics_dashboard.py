"""
Evaluation Metrics Dashboard
============================

Comprehensive evaluation framework tracking:
- Limb activation patterns
- RNA editing effectiveness
- Quantum coherence metrics
- Task-specific performance
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import statistics


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    
    # Limb metrics
    limb_activations: Dict[str, float]  # Which limbs activated
    limb_confidence_avg: float           # Average limb confidence
    limb_entropy: float                  # Diversity of limb usage
    
    # RNA metrics
    rna_pathway_distribution: List[float]  # Which pathways used
    rna_temperature: float                 # Current exploration level
    rna_ei_ratio: float                    # Excitatory/inhibitory balance
    
    # Quantum metrics
    quantum_coherence: float             # Quantum coherence length
    quantum_entanglement: float          # Entanglement strength
    quantum_coupling_eigenvalues: List[float]  # Coupling matrix spectrum
    
    # Task metrics
    task_accuracy: float                 # Accuracy on task
    task_type_detected: str              # Detected task type
    task_confidence: float               # Confidence in classification
    
    # Inference metrics
    inference_time_ms: float             # Total inference time
    stage_times_ms: Dict[str, float]     # Per-stage breakdown


class MetricsDashboard:
    """
    Unified metrics collection and reporting.
    """
    
    def __init__(self):
        self.metrics_history: List[EvaluationMetrics] = []
    
    def extract_from_model_output(
        self,
        model_output: Dict[str, Any],
        task_accuracy: float = 0.0,
        task_type: str = "Unknown",
    ) -> EvaluationMetrics:
        """
        Extract metrics from model output dict.
        """
        metrics_dict = model_output.get('metrics', {})
        
        # Limb metrics
        limb_confidences = model_output.get('limb_confidences', [])
        if limb_confidences:
            limb_conf_avg = float(torch.tensor(limb_confidences).mean())
            limb_entropy = float(self._compute_entropy(limb_confidences))
        else:
            limb_conf_avg = 0.0
            limb_entropy = 0.0
        
        limb_activations = {f"limb_{i}": float(c) for i, c in enumerate(limb_confidences[:8])}
        
        # RNA metrics
        rna_temp = metrics_dict.get('rna_temperature', 1.0)
        rna_conf = metrics_dict.get('rna_confidence', 0.5)
        
        # Quantum metrics
        entanglement = metrics_dict.get('entanglement_strength', 0.0)
        coherence = metrics_dict.get('coherence', 0.0)
        
        # Timing breakdown
        stage_times = {
            'perception': metrics_dict.get('perception_time', 0) * 1000,
            'rna_editing': metrics_dict.get('rna_time', 0) * 1000,
            'limbs': metrics_dict.get('limbs_time', 0) * 1000,
            'quantum': metrics_dict.get('quantum_time', 0) * 1000,
            'reasoning': metrics_dict.get('reasoning_time', 0) * 1000,
            'action': metrics_dict.get('action_time', 0) * 1000,
        }
        
        return EvaluationMetrics(
            limb_activations=limb_activations,
            limb_confidence_avg=limb_conf_avg,
            limb_entropy=limb_entropy,
            rna_pathway_distribution=[],  # TODO: extract from RNA layer
            rna_temperature=float(rna_temp) if isinstance(rna_temp, torch.Tensor) else rna_temp,
            rna_ei_ratio=0.8,  # Target is 80/20
            quantum_coherence=float(coherence) if isinstance(coherence, torch.Tensor) else coherence,
            quantum_entanglement=float(entanglement) if isinstance(entanglement, torch.Tensor) else entanglement,
            quantum_coupling_eigenvalues=[],  # TODO: extract from coupling matrix
            task_accuracy=task_accuracy,
            task_type_detected=task_type,
            task_confidence=rna_conf,
            inference_time_ms=metrics_dict.get('total_time', 0) * 1000,
            stage_times_ms=stage_times,
        )
    
    def add_metrics(self, metrics: EvaluationMetrics):
        """Add metrics to history."""
        self.metrics_history.append(metrics)
    
    def _compute_entropy(self, values: List[float]) -> float:
        """Compute Shannon entropy of normalized values."""
        if not values:
            return 0.0
        
        # Normalize to probabilities
        total = sum(values) + 1e-8
        probs = [v / total for v in values]
        
        # Entropy
        entropy = -sum(p * (math.log(p) if p > 0 else 0) for p in probs)
        return entropy
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all metrics."""
        if not self.metrics_history:
            return {}
        
        # Aggregate across all recorded metrics
        accuracies = [m.task_accuracy for m in self.metrics_history]
        times = [m.inference_time_ms for m in self.metrics_history]
        coherences = [m.quantum_coherence for m in self.metrics_history]
        entanglements = [m.quantum_entanglement for m in self.metrics_history]
        
        return {
            'total_evaluations': len(self.metrics_history),
            'accuracy': {
                'mean': statistics.mean(accuracies) if accuracies else 0,
                'std': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            },
            'inference_time_ms': {
                'mean': statistics.mean(times) if times else 0,
                'std': statistics.stdev(times) if len(times) > 1 else 0,
            },
            'quantum_coherence': {
                'mean': statistics.mean(coherences) if coherences else 0,
                'std': statistics.stdev(coherences) if len(coherences) > 1 else 0,
            },
            'quantum_entanglement': {
                'mean': statistics.mean(entanglements) if entanglements else 0,
                'std': statistics.stdev(entanglements) if len(entanglements) > 1 else 0,
            },
        }
    
    def print_report(self):
        """Print comprehensive metrics report."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("EVALUATION METRICS REPORT")
        print("="*70 + "\n")
        
        print(f"Total Evaluations: {summary.get('total_evaluations', 0)}\n")
        
        print("ACCURACY:")
        print(f"  Mean: {summary['accuracy']['mean']:.1%}")
        print(f"  Std:  {summary['accuracy']['std']:.1%}\n")
        
        print("INFERENCE TIME (ms):")
        print(f"  Mean: {summary['inference_time_ms']['mean']:.1f}")
        print(f"  Std:  {summary['inference_time_ms']['std']:.1f}\n")
        
        print("QUANTUM COHERENCE:")
        print(f"  Mean: {summary['quantum_coherence']['mean']:.3f}")
        print(f"  Std:  {summary['quantum_coherence']['std']:.3f}\n")
        
        print("QUANTUM ENTANGLEMENT:")
        print(f"  Mean: {summary['quantum_entanglement']['mean']:.3f}")
        print(f"  Std:  {summary['quantum_entanglement']['std']:.3f}\n")
        
        print("="*70 + "\n")
    
    def save_report(self, output_path: str = "metrics_report.json"):
        """Save metrics report to JSON."""
        report = {
            'summary': self.get_summary(),
            'detailed_metrics': [asdict(m) for m in self.metrics_history],
        }
        
        with open(output_path, 'w') as f:
            # Convert non-serializable types
            json.dump(report, f, indent=2, default=str)
        
        print(f"Metrics report saved to: {output_path}")


import math

if __name__ == "__main__":
    # Test metrics dashboard
    dashboard = MetricsDashboard()
    
    # Simulate some metrics
    for i in range(5):
        model_output = {
            'metrics': {
                'perception_time': 0.01,
                'rna_time': 0.02,
                'limbs_time': 0.03,
                'quantum_time': 0.01,
                'reasoning_time': 0.02,
                'action_time': 0.01,
                'total_time': 0.1,
                'rna_confidence': 0.8,
                'rna_temperature': 1.0,
                'entanglement_strength': 0.65,
                'coherence': 0.82,
            },
            'limb_confidences': [0.7, 0.8, 0.9, 0.6, 0.75, 0.8, 0.7, 0.65],
        }
        
        metrics = dashboard.extract_from_model_output(
            model_output,
            task_accuracy=0.75 + 0.05 * i,
            task_type="Pattern Completion"
        )
        dashboard.add_metrics(metrics)
    
    dashboard.print_report()
    dashboard.save_report()
    print("✓ Metrics dashboard test complete!")
