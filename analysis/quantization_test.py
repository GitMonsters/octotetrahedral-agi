"""
Quantization Test for OctoTetrahedral AGI

Tests whether the 8-limb architecture exhibits quantum-like quantized activation patterns.

Hypothesis: If the model operates like 8 coupled quantum oscillators, we should see:
1. Discrete peaks in activation histograms (quantized energy levels)
2. Gaussian envelopes around peaks (coherent state signatures)
3. Approximately equal spacing between peaks (harmonic oscillator)
4. Oscillatory patterns in limb activations (Rabi oscillations)

Usage:
    python analysis/quantization_test.py --checkpoint checkpoints/arc/arc_step_2500.pt
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

try:
    import torch
    import torch.nn as nn
    import numpy as np
    from scipy import stats
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    HAS_SCIPY = False
    import torch
    import torch.nn as nn
    import numpy as np

from model import OctoTetrahedralModel
from config import get_config


class QuantizationAnalyzer:
    """
    Analyzes hidden state activations for quantum-like quantization patterns.
    """
    
    def __init__(self, model: OctoTetrahedralModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Storage for activations
        self.limb_activations: Dict[str, List[torch.Tensor]] = {
            'perception': [],
            'memory': [],
            'planning': [],
            'language': [],
            'spatial': [],
            'reasoning': [],
            'metacognition': [],
            'action': []
        }
        
        # Register hooks to capture activations
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture limb activations."""
        self.hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0]
                elif isinstance(output, dict):
                    activation = output.get('hidden_states', output.get('output', None))
                else:
                    activation = output
                    
                if activation is not None and isinstance(activation, torch.Tensor):
                    self.limb_activations[name].append(activation.detach().cpu())
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
    
    def clear_activations(self):
        """Clear stored activations."""
        for key in self.limb_activations:
            self.limb_activations[key] = []
    
    def collect_activations(self, num_samples: int = 100, seq_len: int = 32):
        """
        Run inference on random inputs to collect activation samples.
        """
        print(f"Collecting activations from {num_samples} samples...")
        self.clear_activations()
        
        with torch.no_grad():
            for i in range(num_samples):
                # Random input
                input_ids = torch.randint(
                    0, self.model.vocab_size, 
                    (1, seq_len), 
                    device=self.device
                )
                
                # Forward pass
                _ = self.model(input_ids=input_ids)
                
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{num_samples} samples")
        
        print("Activation collection complete.")
    
    def analyze_quantization(self, limb_name: str) -> Dict:
        """
        Analyze activations from a specific limb for quantization patterns.
        
        Returns:
            Dict with quantization metrics
        """
        activations = self.limb_activations.get(limb_name, [])
        if not activations:
            return {'error': f'No activations found for {limb_name}'}
        
        # Concatenate and flatten
        all_activations = torch.cat(activations, dim=0)
        flat_activations = all_activations.flatten().numpy()
        
        # Basic statistics
        mean_val = np.mean(flat_activations)
        std_val = np.std(flat_activations)
        
        # Histogram analysis
        hist, bin_edges = np.histogram(flat_activations, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks (potential quantized levels)
        if HAS_SCIPY:
            peaks, peak_properties = find_peaks(
                hist, 
                prominence=0.05 * hist.max(),
                distance=5
            )
        else:
            # Simple peak finding without scipy
            peaks = self._simple_find_peaks(hist, threshold=0.05 * hist.max())
            peak_properties = {}
        
        peak_positions = bin_centers[peaks] if len(peaks) > 0 else []
        num_peaks = len(peaks)
        
        # Check for equal spacing (harmonic oscillator signature)
        spacing_regularity = 0.0
        if num_peaks >= 3:
            spacings = np.diff(peak_positions)
            mean_spacing = np.mean(spacings)
            spacing_std = np.std(spacings)
            spacing_regularity = 1.0 - (spacing_std / (mean_spacing + 1e-8))
            spacing_regularity = max(0, min(1, spacing_regularity))
        
        # Gaussian fit quality (coherent state test)
        gaussian_score = self._test_gaussian_peaks(hist, bin_centers, peaks)
        
        # Quantization score: how well does data fit to discrete levels?
        # More peaks around 8 = better match to our hypothesis
        quantization_score = min(1.0, num_peaks / 8.0)
        
        # Overall quantum signature score
        quantum_score = (quantization_score + spacing_regularity + gaussian_score) / 3
        
        return {
            'limb': limb_name,
            'num_samples': len(flat_activations),
            'mean': float(mean_val),
            'std': float(std_val),
            'num_peaks': num_peaks,
            'peak_positions': [float(p) for p in peak_positions],
            'quantization_score': float(quantization_score),
            'spacing_regularity': float(spacing_regularity),
            'gaussian_score': float(gaussian_score),
            'quantum_signature_score': float(quantum_score),
            'histogram': {
                'counts': hist.tolist(),
                'bin_centers': bin_centers.tolist()
            }
        }
    
    def _simple_find_peaks(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Simple peak finding without scipy."""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
                peaks.append(i)
        return np.array(peaks)
    
    def _test_gaussian_peaks(
        self, 
        hist: np.ndarray, 
        bin_centers: np.ndarray, 
        peaks: np.ndarray
    ) -> float:
        """
        Test if peaks have Gaussian shapes (coherent state signature).
        """
        if len(peaks) == 0 or not HAS_SCIPY:
            return 0.5  # Neutral score
        
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        gaussian_scores = []
        for peak_idx in peaks[:8]:  # Analyze up to 8 peaks
            # Extract region around peak
            start = max(0, peak_idx - 10)
            end = min(len(hist), peak_idx + 10)
            
            x_region = bin_centers[start:end]
            y_region = hist[start:end]
            
            if len(x_region) < 5:
                continue
            
            try:
                # Fit Gaussian
                popt, _ = curve_fit(
                    gaussian, x_region, y_region,
                    p0=[hist[peak_idx], bin_centers[peak_idx], 0.1],
                    maxfev=1000
                )
                
                # Calculate R² for fit quality
                y_fit = gaussian(x_region, *popt)
                ss_res = np.sum((y_region - y_fit) ** 2)
                ss_tot = np.sum((y_region - np.mean(y_region)) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                
                gaussian_scores.append(max(0, r_squared))
            except:
                gaussian_scores.append(0.5)
        
        return np.mean(gaussian_scores) if gaussian_scores else 0.5
    
    def analyze_rabi_oscillations(self) -> Dict:
        """
        Test for Rabi oscillation patterns between adjacent limbs.
        """
        limb_order = ['perception', 'memory', 'planning', 'language', 
                      'spatial', 'reasoning', 'metacognition', 'action']
        
        oscillation_scores = []
        
        for i, limb_a in enumerate(limb_order):
            limb_b = limb_order[(i + 1) % 8]
            
            act_a = self.limb_activations.get(limb_a, [])
            act_b = self.limb_activations.get(limb_b, [])
            
            if not act_a or not act_b:
                continue
            
            # Get mean activations over time (samples)
            means_a = [a.mean().item() for a in act_a]
            means_b = [b.mean().item() for b in act_b]
            
            if len(means_a) < 10:
                continue
            
            # Check for anti-correlation (Rabi oscillation signature)
            if HAS_SCIPY:
                correlation, _ = stats.pearsonr(means_a[:len(means_b)], means_b[:len(means_a)])
            else:
                correlation = np.corrcoef(means_a[:len(means_b)], means_b[:len(means_a)])[0, 1]
            
            # Negative correlation = oscillating between limbs
            oscillation_score = max(0, -correlation)
            oscillation_scores.append({
                'limb_pair': f'{limb_a}-{limb_b}',
                'correlation': float(correlation),
                'oscillation_score': float(oscillation_score)
            })
        
        overall_oscillation = np.mean([s['oscillation_score'] for s in oscillation_scores]) if oscillation_scores else 0
        
        return {
            'pairs': oscillation_scores,
            'overall_oscillation_score': float(overall_oscillation)
        }
    
    def run_full_analysis(self, num_samples: int = 100) -> Dict:
        """
        Run complete quantization analysis.
        """
        print("=" * 60)
        print("QUANTUM QUANTIZATION ANALYSIS")
        print("OctoTetrahedral AGI - 8 Coupled Oscillator Test")
        print("=" * 60)
        
        # Collect activations
        self.collect_activations(num_samples=num_samples)
        
        # Analyze each limb
        limb_results = {}
        print("\nAnalyzing limb activations...")
        
        for limb_name in self.limb_activations.keys():
            result = self.analyze_quantization(limb_name)
            limb_results[limb_name] = result
            
            print(f"\n{limb_name.upper()}:")
            print(f"  Peaks found: {result.get('num_peaks', 0)}")
            print(f"  Quantization score: {result.get('quantization_score', 0):.3f}")
            print(f"  Spacing regularity: {result.get('spacing_regularity', 0):.3f}")
            print(f"  Gaussian score: {result.get('gaussian_score', 0):.3f}")
            print(f"  QUANTUM SIGNATURE: {result.get('quantum_signature_score', 0):.3f}")
        
        # Analyze Rabi oscillations
        print("\nAnalyzing Rabi oscillations between limbs...")
        rabi_results = self.analyze_rabi_oscillations()
        
        print(f"\nOverall oscillation score: {rabi_results['overall_oscillation_score']:.3f}")
        for pair in rabi_results['pairs']:
            print(f"  {pair['limb_pair']}: correlation={pair['correlation']:.3f}")
        
        # Overall quantum score
        quantum_scores = [r.get('quantum_signature_score', 0) for r in limb_results.values()]
        overall_quantum = np.mean(quantum_scores) if quantum_scores else 0
        
        combined_score = (overall_quantum + rabi_results['overall_oscillation_score']) / 2
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Average limb quantum signature: {overall_quantum:.3f}")
        print(f"Rabi oscillation score: {rabi_results['overall_oscillation_score']:.3f}")
        print(f"COMBINED QUANTUM SCORE: {combined_score:.3f}")
        print()
        
        if combined_score > 0.6:
            print("RESULT: STRONG evidence of quantum-like behavior")
            print("The 8-limb architecture shows quantized activation patterns!")
        elif combined_score > 0.4:
            print("RESULT: MODERATE evidence of quantum-like behavior")
            print("Some quantization present, may need more training data.")
        else:
            print("RESULT: WEAK evidence of quantum-like behavior")
            print("Activations appear more continuous than quantized.")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'num_samples': num_samples,
            'limb_analysis': limb_results,
            'rabi_oscillations': rabi_results,
            'overall_quantum_score': float(overall_quantum),
            'overall_oscillation_score': float(rabi_results['overall_oscillation_score']),
            'combined_quantum_score': float(combined_score)
        }
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description='Test for quantized activations in OctoTetrahedral AGI')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/arc/arc_step_2500.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of inference samples to collect')
    parser.add_argument('--output', type=str, default='analysis/quantization_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    
    try:
        model, checkpoint_info = OctoTetrahedralModel.load_checkpoint(
            args.checkpoint, 
            device=args.device
        )
        print(f"Model loaded successfully!")
        print(f"  Parameters: {model.get_num_params():,}")
        print(f"  Step: {checkpoint_info.get('step', 'unknown')}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Creating fresh model for testing...")
        config = get_config()
        model = OctoTetrahedralModel(config)
    
    # Run analysis
    analyzer = QuantizationAnalyzer(model, device=args.device)
    
    try:
        results = analyzer.run_full_analysis(num_samples=args.num_samples)
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            # Remove histogram data for cleaner output (too large)
            results_clean = {k: v for k, v in results.items()}
            for limb in results_clean.get('limb_analysis', {}).values():
                if 'histogram' in limb:
                    del limb['histogram']
            json.dump(results_clean, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        
    finally:
        analyzer.cleanup()
    
    return results


if __name__ == '__main__':
    main()
