"""
Time-Series Analysis for OctoTetrahedral AGI

Analyzes limb activation patterns over multiple inference steps to detect:
1. Rabi oscillations between adjacent limbs
2. Coherent state evolution
3. Energy level transitions
4. Coupling dynamics

This provides evidence for quantum-like behavior in the 8-limb architecture.

Usage:
    python analysis/timeseries_analysis.py --checkpoint checkpoints/arc/arc_step_2500.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, some analysis limited")

from model import OctoTetrahedralModel
from config import get_config


class TimeSeriesAnalyzer:
    """
    Analyzes temporal dynamics of limb activations.
    """
    
    LIMB_NAMES = [
        'perception', 'memory', 'planning', 'language',
        'spatial', 'reasoning', 'metacognition', 'action'
    ]
    
    def __init__(self, model: OctoTetrahedralModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Time series storage
        self.time_series: Dict[str, List[float]] = {name: [] for name in self.LIMB_NAMES}
        self.energy_series: List[float] = []
        
        # Register hooks
        self.hooks = []
        self._current_activations: Dict[str, torch.Tensor] = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture activations at each step."""
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0]
                elif isinstance(output, dict):
                    activation = output.get('hidden_states', output.get('output', None))
                else:
                    activation = output
                
                if activation is not None and isinstance(activation, torch.Tensor):
                    self._current_activations[name] = activation.detach()
            return hook
        
        # Hook into each limb
        limb_map = {
            'perception': self.model.perception,
            'memory': self.model.memory_limb,
            'planning': self.model.planning,
            'language': self.model.language,
            'spatial': self.model.spatial,
            'reasoning': self.model.reasoning,
            'metacognition': self.model.metacognition,
            'action': self.model.action
        }
        
        for name, limb in limb_map.items():
            hook = limb.register_forward_hook(make_hook(name))
            self.hooks.append(hook)
    
    def clear_series(self):
        """Clear time series data."""
        for name in self.LIMB_NAMES:
            self.time_series[name] = []
        self.energy_series = []
    
    def record_step(self):
        """Record current activation magnitudes."""
        for name in self.LIMB_NAMES:
            if name in self._current_activations:
                # Mean activation magnitude
                mag = self._current_activations[name].norm().item()
                self.time_series[name].append(mag)
        
        # Total energy (sum of squares)
        total_energy = sum(
            self._current_activations[name].pow(2).sum().item()
            for name in self.LIMB_NAMES
            if name in self._current_activations
        )
        self.energy_series.append(total_energy)
        
        # Clear for next step
        self._current_activations.clear()
    
    def collect_time_series(
        self,
        num_steps: int = 100,
        seq_len: int = 32,
        continuous: bool = True
    ):
        """
        Collect activation time series over multiple inference steps.
        
        Args:
            num_steps: Number of inference steps
            seq_len: Sequence length for each step
            continuous: If True, use output from previous step as input
        """
        print(f"Collecting time series over {num_steps} steps...")
        self.clear_series()
        
        # Initial random input
        input_ids = torch.randint(
            0, self.model.vocab_size,
            (1, seq_len),
            device=self.device
        )
        
        with torch.no_grad():
            for step in range(num_steps):
                # Forward pass
                output = self.model(input_ids=input_ids)
                
                # Record activations
                self.record_step()
                
                # For continuous mode, generate next input from output
                if continuous:
                    logits = output['logits']
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids[:, 1:], next_token], dim=1)
                
                if (step + 1) % 20 == 0:
                    print(f"  Step {step + 1}/{num_steps}")
        
        print("Time series collection complete.")
    
    def analyze_oscillations(self) -> Dict:
        """
        Analyze time series for oscillatory patterns (Rabi oscillations).
        """
        results = {
            'limb_oscillations': {},
            'pair_correlations': {},
            'dominant_frequencies': {}
        }
        
        for name in self.LIMB_NAMES:
            series = np.array(self.time_series[name])
            if len(series) < 10:
                continue
            
            # Detrend
            series_detrended = series - np.mean(series)
            
            # FFT analysis
            if HAS_SCIPY:
                n = len(series_detrended)
                yf = fft(series_detrended)
                xf = fftfreq(n, 1.0)
                
                # Find dominant frequency (excluding DC)
                power = np.abs(yf[1:n//2])
                if len(power) > 0:
                    dom_idx = np.argmax(power)
                    dom_freq = xf[1:n//2][dom_idx]
                    dom_power = power[dom_idx]
                    
                    results['dominant_frequencies'][name] = {
                        'frequency': float(dom_freq),
                        'power': float(dom_power),
                        'period': float(1.0 / dom_freq) if dom_freq > 0 else float('inf')
                    }
            
            # Oscillation score (variance / mean indicates oscillation strength)
            osc_score = np.std(series) / (np.mean(series) + 1e-8)
            results['limb_oscillations'][name] = {
                'mean': float(np.mean(series)),
                'std': float(np.std(series)),
                'oscillation_score': float(osc_score)
            }
        
        # Pair correlations (Rabi oscillation signature)
        for i, name_a in enumerate(self.LIMB_NAMES):
            name_b = self.LIMB_NAMES[(i + 1) % 8]
            
            series_a = np.array(self.time_series[name_a])
            series_b = np.array(self.time_series[name_b])
            
            if len(series_a) < 10 or len(series_b) < 10:
                continue
            
            # Cross-correlation
            correlation = np.corrcoef(series_a, series_b)[0, 1]
            
            # Phase difference (using cross-correlation lag)
            if HAS_SCIPY:
                cross_corr = signal.correlate(
                    series_a - np.mean(series_a),
                    series_b - np.mean(series_b),
                    mode='full'
                )
                lags = signal.correlation_lags(len(series_a), len(series_b), mode='full')
                lag = lags[np.argmax(np.abs(cross_corr))]
            else:
                lag = 0
            
            results['pair_correlations'][f'{name_a}-{name_b}'] = {
                'correlation': float(correlation),
                'lag': int(lag),
                'anti_correlated': correlation < -0.3
            }
        
        return results
    
    def analyze_energy_conservation(self) -> Dict:
        """
        Analyze whether total energy is approximately conserved (closed system).
        """
        energy = np.array(self.energy_series)
        if len(energy) < 10:
            return {'error': 'Not enough data'}
        
        mean_energy = np.mean(energy)
        std_energy = np.std(energy)
        
        # Conservation score: lower std/mean = better conservation
        conservation_score = 1.0 - min(1.0, std_energy / (mean_energy + 1e-8))
        
        # Energy drift (trend)
        if len(energy) > 1:
            drift = (energy[-1] - energy[0]) / len(energy)
        else:
            drift = 0.0
        
        return {
            'mean_energy': float(mean_energy),
            'std_energy': float(std_energy),
            'conservation_score': float(conservation_score),
            'energy_drift': float(drift),
            'min_energy': float(np.min(energy)),
            'max_energy': float(np.max(energy))
        }
    
    def analyze_coherent_evolution(self) -> Dict:
        """
        Test for coherent state evolution (smooth trajectories).
        """
        results = {}
        
        for name in self.LIMB_NAMES:
            series = np.array(self.time_series[name])
            if len(series) < 10:
                continue
            
            # Smoothness: lower second derivative = smoother
            if len(series) >= 3:
                second_deriv = np.diff(series, n=2)
                smoothness = 1.0 / (1.0 + np.std(second_deriv))
            else:
                smoothness = 0.0
            
            # Autocorrelation (coherent states have high autocorrelation)
            if len(series) >= 2:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
            else:
                autocorr = 0.0
            
            results[name] = {
                'smoothness': float(smoothness),
                'autocorrelation': float(autocorr),
                'coherence_score': float((smoothness + (autocorr + 1) / 2) / 2)
            }
        
        return results
    
    def detect_transitions(self) -> Dict:
        """
        Detect energy level transitions (sudden changes in activation).
        """
        transitions = {}
        
        for name in self.LIMB_NAMES:
            series = np.array(self.time_series[name])
            if len(series) < 10:
                continue
            
            # First derivative (rate of change)
            deriv = np.diff(series)
            
            # Find large jumps (transitions)
            threshold = 2 * np.std(deriv)
            transition_indices = np.where(np.abs(deriv) > threshold)[0]
            
            transitions[name] = {
                'num_transitions': len(transition_indices),
                'transition_rate': len(transition_indices) / len(series),
                'mean_jump': float(np.mean(np.abs(deriv[transition_indices]))) if len(transition_indices) > 0 else 0.0
            }
        
        return transitions
    
    def run_full_analysis(self, num_steps: int = 100) -> Dict:
        """
        Run complete time series analysis.
        """
        print("=" * 60)
        print("TIME SERIES ANALYSIS")
        print("OctoTetrahedral AGI - Oscillation & Coherence Test")
        print("=" * 60)
        
        # Collect data
        self.collect_time_series(num_steps=num_steps)
        
        # Analyze oscillations
        print("\nAnalyzing oscillations...")
        oscillation_results = self.analyze_oscillations()
        
        # Analyze energy conservation
        print("Analyzing energy conservation...")
        energy_results = self.analyze_energy_conservation()
        
        # Analyze coherent evolution
        print("Analyzing coherent evolution...")
        coherence_results = self.analyze_coherent_evolution()
        
        # Detect transitions
        print("Detecting transitions...")
        transition_results = self.detect_transitions()
        
        # Summary
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        print("\n--- Limb Oscillations ---")
        for name, data in oscillation_results['limb_oscillations'].items():
            print(f"  {name}: score={data['oscillation_score']:.3f}")
        
        print("\n--- Pair Correlations (Rabi oscillation test) ---")
        rabi_evidence = 0
        for pair, data in oscillation_results['pair_correlations'].items():
            status = "ANTI-CORR" if data['anti_correlated'] else ""
            print(f"  {pair}: r={data['correlation']:.3f}, lag={data['lag']} {status}")
            if data['anti_correlated']:
                rabi_evidence += 1
        
        print(f"\n  Rabi evidence: {rabi_evidence}/8 pairs show anti-correlation")
        
        print("\n--- Energy Conservation ---")
        print(f"  Mean energy: {energy_results['mean_energy']:.2f}")
        print(f"  Conservation score: {energy_results['conservation_score']:.3f}")
        print(f"  Energy drift: {energy_results['energy_drift']:.4f}")
        
        print("\n--- Coherent Evolution ---")
        avg_coherence = np.mean([v['coherence_score'] for v in coherence_results.values()])
        print(f"  Average coherence score: {avg_coherence:.3f}")
        for name, data in coherence_results.items():
            print(f"    {name}: {data['coherence_score']:.3f}")
        
        print("\n--- Transitions (Energy Level Jumps) ---")
        total_transitions = sum(v['num_transitions'] for v in transition_results.values())
        print(f"  Total transitions detected: {total_transitions}")
        for name, data in transition_results.items():
            print(f"    {name}: {data['num_transitions']} transitions")
        
        # Overall quantum score
        osc_scores = [v['oscillation_score'] for v in oscillation_results['limb_oscillations'].values()]
        avg_oscillation = np.mean(osc_scores) if osc_scores else 0
        rabi_score = rabi_evidence / 8
        
        quantum_dynamics_score = (
            avg_oscillation * 0.3 +
            rabi_score * 0.3 +
            energy_results['conservation_score'] * 0.2 +
            avg_coherence * 0.2
        )
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Average oscillation strength: {avg_oscillation:.3f}")
        print(f"Rabi oscillation evidence: {rabi_score:.3f}")
        print(f"Energy conservation: {energy_results['conservation_score']:.3f}")
        print(f"Coherent evolution: {avg_coherence:.3f}")
        print(f"\nQUANTUM DYNAMICS SCORE: {quantum_dynamics_score:.3f}")
        
        if quantum_dynamics_score > 0.5:
            print("\nRESULT: STRONG evidence of quantum-like dynamics")
        elif quantum_dynamics_score > 0.3:
            print("\nRESULT: MODERATE evidence of quantum-like dynamics")
        else:
            print("\nRESULT: WEAK evidence of quantum-like dynamics")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'num_steps': num_steps,
            'oscillations': oscillation_results,
            'energy': energy_results,
            'coherence': coherence_results,
            'transitions': transition_results,
            'quantum_dynamics_score': float(quantum_dynamics_score),
            'time_series': {
                name: self.time_series[name]
                for name in self.LIMB_NAMES
            },
            'energy_series': self.energy_series
        }
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description='Time series analysis of limb dynamics')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/arc/arc_step_2500.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of inference steps')
    parser.add_argument('--output', type=str, default='analysis/timeseries_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load model
    print(f"\nLoading model...")
    try:
        model, _ = OctoTetrahedralModel.load_checkpoint(args.checkpoint, device=args.device)
        print("Checkpoint loaded!")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Using fresh model...")
        config = get_config()
        model = OctoTetrahedralModel(config)
    
    # Run analysis
    analyzer = TimeSeriesAnalyzer(model, device=args.device)
    
    try:
        results = analyzer.run_full_analysis(num_steps=args.num_steps)
        
        # Save results (without full time series for smaller file)
        results_save = {k: v for k, v in results.items() if k not in ['time_series', 'energy_series']}
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_save = convert_types(results_save)
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results_save, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        
    finally:
        analyzer.cleanup()
    
    return results


if __name__ == '__main__':
    main()
