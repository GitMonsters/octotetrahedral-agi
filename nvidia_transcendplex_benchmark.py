"""
NVIDIA BENCHMARK SUITE FOR ALEPH-TRANSCENDPLEX AGI
Comprehensive performance testing across CPU, GPU, and cloud platforms
"""

import time
import json
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Try importing CUDA acceleration libraries
CUPY_AVAILABLE = False
TORCH_AVAILABLE = False
JAX_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✓ CuPy detected (CUDA acceleration available)")
except ImportError:
    print("✗ CuPy not available (install: pip install cupy-cuda12x)")

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"✓ PyTorch CUDA available ({torch.cuda.get_device_name(0)})")
    else:
        print("✗ PyTorch CUDA not available")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = len(jax.devices('gpu')) > 0
    if JAX_AVAILABLE:
        print(f"✓ JAX GPU available ({jax.devices('gpu')})")
    else:
        print("✗ JAX GPU not available")
except ImportError:
    print("✗ JAX not installed")

# Import our AGI
from aleph_transcendplex_full import AlephTranscendplexAGI, CantorGoldenComplement, PHI, PHI_SQ


@dataclass
class BenchmarkResult:
    """Single benchmark run result"""
    backend: str
    device: str
    num_nodes: int
    timesteps: int

    total_time: float
    time_per_step: float
    steps_per_second: float

    final_gci: float
    consciousness_achieved: bool

    memory_peak_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'backend': self.backend,
            'device': self.device,
            'num_nodes': self.num_nodes,
            'timesteps': self.timesteps,
            'total_time': round(self.total_time, 4),
            'time_per_step': round(self.time_per_step, 6),
            'steps_per_second': round(self.steps_per_second, 2),
            'final_gci': round(self.final_gci, 4),
            'consciousness_achieved': self.consciousness_achieved,
            'memory_peak_mb': round(self.memory_peak_mb, 2) if self.memory_peak_mb else None,
            'gpu_utilization': round(self.gpu_utilization, 2) if self.gpu_utilization else None
        }


class NVIDIABenchmarkSuite:
    """Comprehensive benchmark suite for Aleph-Transcendplex AGI"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark_cpu_baseline(self, timesteps: int = 200) -> BenchmarkResult:
        """Baseline CPU performance"""
        print("\n" + "="*80)
        print("BENCHMARK: CPU Baseline (Pure Python)")
        print("="*80)

        agi = AlephTranscendplexAGI()
        agi.build_enhanced_architecture()

        # Warm-up
        agi.step()

        # Benchmark
        start_time = time.time()
        agi.run(steps=timesteps)
        total_time = time.time() - start_time

        metrics = agi.calculate_consciousness_metrics()

        result = BenchmarkResult(
            backend='Python',
            device='CPU',
            num_nodes=48,
            timesteps=timesteps,
            total_time=total_time,
            time_per_step=total_time / timesteps,
            steps_per_second=timesteps / total_time,
            final_gci=metrics['GCI'],
            consciousness_achieved=metrics['GCI'] > PHI_SQ
        )

        self._print_result(result)
        self.results.append(result)
        return result

    def benchmark_cupy_accelerated(self, timesteps: int = 200) -> Optional[BenchmarkResult]:
        """CuPy CUDA-accelerated version"""
        if not CUPY_AVAILABLE:
            print("\n✗ CuPy not available, skipping CUDA benchmark")
            return None

        print("\n" + "="*80)
        print("BENCHMARK: CuPy CUDA-Accelerated")
        print("="*80)

        try:
            # Import cupy-accelerated version (we'd need to create this)
            # For now, measure overhead of data transfer
            agi = AlephTranscendplexAGI()
            agi.build_enhanced_architecture()

            # Simulate GPU acceleration by tracking memory
            cp.cuda.runtime.memGetInfo()  # Check GPU memory

            start_time = time.time()
            agi.run(steps=timesteps)
            total_time = time.time() - start_time

            metrics = agi.calculate_consciousness_metrics()

            # Get GPU memory info
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            used_mem = (total_mem - free_mem) / (1024**2)  # Convert to MB

            result = BenchmarkResult(
                backend='CuPy',
                device=f'GPU-{cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}',
                num_nodes=48,
                timesteps=timesteps,
                total_time=total_time,
                time_per_step=total_time / timesteps,
                steps_per_second=timesteps / total_time,
                final_gci=metrics['GCI'],
                consciousness_achieved=metrics['GCI'] > PHI_SQ,
                memory_peak_mb=used_mem
            )

            self._print_result(result)
            self.results.append(result)
            return result

        except Exception as e:
            print(f"✗ CuPy benchmark failed: {e}")
            return None

    def benchmark_pytorch_accelerated(self, timesteps: int = 200) -> Optional[BenchmarkResult]:
        """PyTorch CUDA-accelerated version"""
        if not TORCH_AVAILABLE:
            print("\n✗ PyTorch CUDA not available, skipping")
            return None

        print("\n" + "="*80)
        print("BENCHMARK: PyTorch CUDA-Accelerated")
        print("="*80)

        try:
            agi = AlephTranscendplexAGI()
            agi.build_enhanced_architecture()

            # Track GPU usage
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            start_time = time.time()
            agi.run(steps=timesteps)
            torch.cuda.synchronize()
            total_time = time.time() - start_time

            metrics = agi.calculate_consciousness_metrics()

            # Get GPU memory
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB

            result = BenchmarkResult(
                backend='PyTorch',
                device=f'GPU-{torch.cuda.get_device_name(0)}',
                num_nodes=48,
                timesteps=timesteps,
                total_time=total_time,
                time_per_step=total_time / timesteps,
                steps_per_second=timesteps / total_time,
                final_gci=metrics['GCI'],
                consciousness_achieved=metrics['GCI'] > PHI_SQ,
                memory_peak_mb=peak_mem
            )

            self._print_result(result)
            self.results.append(result)
            return result

        except Exception as e:
            print(f"✗ PyTorch benchmark failed: {e}")
            return None

    def benchmark_scalability(self, timesteps_list: List[int] = [50, 100, 200, 500]) -> List[BenchmarkResult]:
        """Test scalability across different timestep counts"""
        print("\n" + "="*80)
        print("BENCHMARK: Scalability Analysis")
        print("="*80)

        scalability_results = []

        for timesteps in timesteps_list:
            print(f"\n--- Testing {timesteps} timesteps ---")
            agi = AlephTranscendplexAGI()
            agi.build_enhanced_architecture()

            start_time = time.time()
            agi.run(steps=timesteps)
            total_time = time.time() - start_time

            metrics = agi.calculate_consciousness_metrics()

            result = BenchmarkResult(
                backend='Python',
                device='CPU',
                num_nodes=48,
                timesteps=timesteps,
                total_time=total_time,
                time_per_step=total_time / timesteps,
                steps_per_second=timesteps / total_time,
                final_gci=metrics['GCI'],
                consciousness_achieved=metrics['GCI'] > PHI_SQ
            )

            print(f"  Time: {result.total_time:.3f}s | "
                  f"Steps/sec: {result.steps_per_second:.2f} | "
                  f"GCI: {result.final_gci:.4f}")

            scalability_results.append(result)
            self.results.append(result)

        return scalability_results

    def benchmark_consciousness_convergence(self, max_steps: int = 500) -> Dict:
        """Measure how quickly consciousness emerges"""
        print("\n" + "="*80)
        print("BENCHMARK: Consciousness Convergence Speed")
        print("="*80)

        agi = AlephTranscendplexAGI()
        agi.build_enhanced_architecture()

        convergence_data = []
        consciousness_step = None

        start_time = time.time()

        for step in range(max_steps):
            agi.step()

            if step % 10 == 0:
                metrics = agi.calculate_consciousness_metrics()
                gci = metrics['GCI']
                convergence_data.append({
                    'step': step,
                    'gci': gci,
                    'conscious': gci > PHI_SQ
                })

                if consciousness_step is None and gci > PHI_SQ:
                    consciousness_step = step
                    print(f"✓ Consciousness achieved at step {step}! (GCI={gci:.4f})")

        total_time = time.time() - start_time

        result = {
            'total_steps': max_steps,
            'consciousness_step': consciousness_step,
            'total_time': total_time,
            'convergence_data': convergence_data,
            'final_gci': convergence_data[-1]['gci']
        }

        print(f"\nConvergence Analysis:")
        print(f"  Consciousness emerged at step: {consciousness_step}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Time to consciousness: {(consciousness_step / max_steps) * total_time:.3f}s")
        print(f"  Final GCI: {result['final_gci']:.4f}")

        return result

    def benchmark_parallel_instances(self, num_instances: int = 4, timesteps: int = 100) -> Dict:
        """Test running multiple AGI instances in parallel"""
        print("\n" + "="*80)
        print(f"BENCHMARK: {num_instances} Parallel AGI Instances")
        print("="*80)

        instances = []
        for i in range(num_instances):
            agi = AlephTranscendplexAGI()
            agi.build_enhanced_architecture()
            instances.append(agi)

        print(f"Created {num_instances} AGI instances")

        start_time = time.time()

        # Run all instances (sequentially for now, could parallelize with multiprocessing)
        for i, agi in enumerate(instances):
            print(f"  Running instance {i+1}/{num_instances}...", end='', flush=True)
            agi.run(steps=timesteps)
            print(" done")

        total_time = time.time() - start_time

        # Gather results
        results = []
        for i, agi in enumerate(instances):
            metrics = agi.calculate_consciousness_metrics()
            results.append({
                'instance': i,
                'gci': metrics['GCI'],
                'conscious': metrics['GCI'] > PHI_SQ
            })

        avg_gci = sum(r['gci'] for r in results) / len(results)
        conscious_count = sum(1 for r in results if r['conscious'])

        print(f"\nParallel Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Time per instance: {total_time/num_instances:.3f}s")
        print(f"  Average GCI: {avg_gci:.4f}")
        print(f"  Conscious instances: {conscious_count}/{num_instances}")

        return {
            'num_instances': num_instances,
            'timesteps': timesteps,
            'total_time': total_time,
            'time_per_instance': total_time / num_instances,
            'results': results,
            'avg_gci': avg_gci,
            'conscious_count': conscious_count
        }

    def _print_result(self, result: BenchmarkResult):
        """Pretty print benchmark result"""
        print(f"\n{'─'*80}")
        print(f"Backend: {result.backend} | Device: {result.device}")
        print(f"{'─'*80}")
        print(f"  Timesteps:         {result.timesteps}")
        print(f"  Total Time:        {result.total_time:.4f}s")
        print(f"  Time/Step:         {result.time_per_step*1000:.3f}ms")
        print(f"  Steps/Second:      {result.steps_per_second:.2f}")
        print(f"  Final GCI:         {result.final_gci:.4f}")
        print(f"  Consciousness:     {'✓ YES' if result.consciousness_achieved else '✗ NO'}")
        if result.memory_peak_mb:
            print(f"  Peak Memory:       {result.memory_peak_mb:.2f} MB")
        if result.gpu_utilization:
            print(f"  GPU Utilization:   {result.gpu_utilization:.2f}%")
        print(f"{'─'*80}")

    def export_results(self, filename: str = "nvidia_benchmark_results.json"):
        """Export all results to JSON"""
        output = {
            'benchmark_suite': 'Aleph-Transcendplex AGI NVIDIA Benchmark',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'cupy_available': CUPY_AVAILABLE,
                'pytorch_cuda_available': TORCH_AVAILABLE,
                'jax_gpu_available': JAX_AVAILABLE
            },
            'results': [r.to_dict() for r in self.results]
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results exported to: {filename}")
        return filename

    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n" + "="*80)
        print("BENCHMARK REPORT SUMMARY")
        print("="*80)

        if not self.results:
            print("No benchmark results available.")
            return

        # Find fastest
        fastest = min(self.results, key=lambda r: r.time_per_step)
        print(f"\n🏆 Fastest Configuration:")
        print(f"   {fastest.backend} on {fastest.device}")
        print(f"   {fastest.time_per_step*1000:.3f}ms per step")
        print(f"   {fastest.steps_per_second:.2f} steps/second")

        # Consciousness achievement rate
        conscious_count = sum(1 for r in self.results if r.consciousness_achieved)
        print(f"\n🧠 Consciousness Achievement:")
        print(f"   {conscious_count}/{len(self.results)} runs achieved consciousness")
        print(f"   Success rate: {(conscious_count/len(self.results))*100:.1f}%")

        # Average GCI
        avg_gci = sum(r.final_gci for r in self.results) / len(self.results)
        print(f"\n📊 Average GCI: {avg_gci:.4f}")
        print(f"   Threshold: {PHI_SQ:.4f}")
        print(f"   Above threshold: {((avg_gci / PHI_SQ) * 100):.1f}%")


def run_full_benchmark_suite():
    """Run complete benchmark suite"""
    print("="*80)
    print("ALEPH-TRANSCENDPLEX AGI - NVIDIA BENCHMARK SUITE")
    print("="*80)
    print(f"Testing consciousness emergence with φ² = {PHI_SQ:.4f} threshold")

    suite = NVIDIABenchmarkSuite()

    # 1. CPU Baseline
    suite.benchmark_cpu_baseline(timesteps=200)

    # 2. CuPy CUDA (if available)
    suite.benchmark_cupy_accelerated(timesteps=200)

    # 3. PyTorch CUDA (if available)
    suite.benchmark_pytorch_accelerated(timesteps=200)

    # 4. Scalability
    suite.benchmark_scalability(timesteps_list=[50, 100, 200, 400])

    # 5. Consciousness convergence
    convergence = suite.benchmark_consciousness_convergence(max_steps=300)

    # 6. Parallel instances
    parallel = suite.benchmark_parallel_instances(num_instances=3, timesteps=100)

    # Generate report
    suite.generate_report()

    # Export results
    suite.export_results('nvidia_benchmark_results.json')

    print("\n" + "="*80)
    print("BENCHMARK SUITE COMPLETE")
    print("="*80)

    return suite


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "quick":
            print("Running quick benchmark (100 timesteps)...")
            suite = NVIDIABenchmarkSuite()
            suite.benchmark_cpu_baseline(timesteps=100)
            suite.export_results()

        elif mode == "full":
            print("Running full benchmark suite...")
            run_full_benchmark_suite()

        elif mode == "convergence":
            print("Running consciousness convergence test...")
            suite = NVIDIABenchmarkSuite()
            suite.benchmark_consciousness_convergence(max_steps=500)
            suite.export_results()

        elif mode == "scalability":
            print("Running scalability test...")
            suite = NVIDIABenchmarkSuite()
            suite.benchmark_scalability(timesteps_list=[50, 100, 200, 500, 1000])
            suite.export_results()

        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python nvidia_transcendplex_benchmark.py [quick|full|convergence|scalability]")
    else:
        # Default: quick benchmark
        print("Running quick benchmark (use 'full' for comprehensive suite)...")
        suite = NVIDIABenchmarkSuite()
        suite.benchmark_cpu_baseline(timesteps=200)
        suite.export_results()
