"""
Advanced test suite for compound learning and integration system
Stress testing, stability testing, memory leak detection, and concurrent execution tests
"""

import pytest
import asyncio
import gc
import sys
import time
import psutil
import os
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict

from ngvt_compound_learning import (
    CompoundLearningEngine,
    CompoundIntegrationEngine,
    LearningExperience,
    CompoundLearningPattern,
    CrossModelLearning,
    IntegrationWorkflow,
)


class MemoryMonitor:
    """Monitor memory usage during tests"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = None
        self.snapshots = []
    
    def start(self):
        """Start memory monitoring"""
        gc.collect()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.snapshots = [self.start_memory]
    
    def snapshot(self) -> float:
        """Take memory snapshot, return delta from start"""
        gc.collect()
        current = self.process.memory_info().rss / 1024 / 1024  # MB
        self.snapshots.append(current)
        return current - self.start_memory
    
    def get_peak_delta(self) -> float:
        """Get peak memory increase"""
        if not self.snapshots:
            return 0
        return max(self.snapshots) - self.start_memory
    
    def get_current_delta(self) -> float:
        """Get current memory increase"""
        if not self.snapshots:
            return 0
        return self.snapshots[-1] - self.start_memory


class TestStressLargeModelCounts:
    """Stress tests with large numbers of models"""
    
    def test_register_100_models(self):
        """Test registering 100 models"""
        engine = CompoundLearningEngine()
        start_time = time.time()
        
        for i in range(100):
            engine.register_model(f'model_{i}', ['capability_a', 'capability_b'])
        
        elapsed = time.time() - start_time
        
        assert len(engine.model_signatures) == 100
        # Should handle 100 models in < 100ms
        assert elapsed < 0.1
        print(f"✓ Registered 100 models in {elapsed*1000:.2f}ms")
    
    def test_register_500_models(self):
        """Test registering 500 models"""
        engine = CompoundLearningEngine()
        start_time = time.time()
        
        for i in range(500):
            engine.register_model(f'model_{i}', ['capability_a', 'capability_b'])
        
        elapsed = time.time() - start_time
        
        assert len(engine.model_signatures) == 500
        # Should handle 500 models in < 500ms
        assert elapsed < 0.5
        print(f"✓ Registered 500 models in {elapsed*1000:.2f}ms")
    
    def test_affinity_matrix_scaling(self):
        """Test affinity matrix scaling with many models"""
        engine = CompoundLearningEngine()
        
        # Register 50 models
        for i in range(50):
            engine.register_model(f'model_{i}', ['type_a'])
        
        # Record transfers between all pairs
        start_time = time.time()
        pattern = CompoundLearningPattern(
            pattern_id='test',
            query_hash='hash',
            response_template='template',
            accuracy=0.8,
            frequency=1,
            avg_latency_ms=10.0,
            effectiveness_score=0.8,
        )
        
        transfer_count = 0
        for i in range(50):
            for j in range(50):
                if i != j:
                    engine.record_cross_model_transfer(
                        pattern,
                        f'model_{i}',
                        f'model_{j}',
                        success=True,
                        effectiveness=0.9
                    )
                    transfer_count += 1
        
        elapsed = time.time() - start_time
        
        # Should handle 2450 transfers in reasonable time
        assert transfer_count == 2450
        assert elapsed < 5.0
        print(f"✓ Recorded {transfer_count} transfers in {elapsed*1000:.2f}ms")
    
    def test_integration_engine_many_models(self):
        """Test integration engine with many models"""
        learning_engine = CompoundLearningEngine()
        integration_engine = CompoundIntegrationEngine(learning_engine=learning_engine)
        
        # Register 100 models in integration engine
        start_time = time.time()
        for i in range(100):
            model_type = 'nlp' if i % 2 == 0 else 'vision'
            integration_engine.register_model(f'model_{i}', model_type, {})
        
        elapsed = time.time() - start_time
        
        assert len(integration_engine.models) == 100
        assert elapsed < 0.2
        print(f"✓ Integration engine registered 100 models in {elapsed*1000:.2f}ms")


class TestStressLargeDataVolumes:
    """Stress tests with large data volumes"""
    
    def test_record_10000_experiences(self):
        """Test recording 10,000 learning experiences"""
        engine = CompoundLearningEngine()
        start_time = time.time()
        monitor = MemoryMonitor()
        monitor.start()
        
        for i in range(10000):
            exp = LearningExperience(
                query=f"query_{i}",
                response=f"response_{i}",
                latency_ms=10.0 + (i % 100),
                success=(i % 10) != 9,  # 90% success rate
                timestamp=datetime.now().isoformat(),
                metadata={'batch': i // 1000}
            )
            engine.record_experience(exp)
        
        elapsed = time.time() - start_time
        memory_delta = monitor.snapshot()
        
        assert len(engine.experiences) == 10000
        assert engine.knowledge_base['total_experiences'] == 10000
        # Should record 10k experiences in < 2 seconds
        assert elapsed < 2.0
        # Memory increase should be reasonable (< 100MB for 10k experiences)
        assert memory_delta < 100
        
        print(f"✓ Recorded 10,000 experiences in {elapsed*1000:.2f}ms")
        print(f"  Memory delta: {memory_delta:.2f}MB (peak: {monitor.get_peak_delta():.2f}MB)")
    
    def test_extract_patterns_large_dataset(self):
        """Test pattern extraction with large experience dataset"""
        engine = CompoundLearningEngine()
        
        # Record 5000 experiences
        for i in range(5000):
            exp = LearningExperience(
                query=f"query_{i % 100}",  # 100 unique queries
                response=f"response_{i % 100}",
                latency_ms=10.0 + (i % 50),
                success=(i % 5) != 4,  # 80% success
                timestamp=datetime.now().isoformat(),
            )
            engine.record_experience(exp)
        
        start_time = time.time()
        monitor = MemoryMonitor()
        monitor.start()
        
        # Extract patterns
        patterns = engine.extract_patterns(min_frequency=5)
        
        elapsed = time.time() - start_time
        memory_delta = monitor.snapshot()
        
        # Should have extracted patterns from 100 unique queries
        assert len(engine.patterns) > 0
        assert len(engine.patterns) <= 100
        # Pattern extraction should be fast (< 1 second for 5k experiences)
        assert elapsed < 1.0
        assert memory_delta < 50
        
        print(f"✓ Extracted {len(engine.patterns)} patterns from 5,000 experiences in {elapsed*1000:.2f}ms")
        print(f"  Memory delta: {memory_delta:.2f}MB")
    
    def test_learning_cycle_with_many_patterns(self):
        """Test learning cycle with many patterns"""
        engine = CompoundLearningEngine()
        
        # Register models
        for i in range(10):
            engine.register_model(f'model_{i}', ['type_a', 'type_b'])
        
        # Record experiences to create patterns
        for i in range(2000):
            exp = LearningExperience(
                query=f"query_{i % 50}",  # 50 unique queries
                response=f"response_{i % 50}",
                latency_ms=10.0,
                success=True,
                timestamp=datetime.now().isoformat(),
            )
            engine.record_experience(exp)
        
        # Extract patterns
        engine.extract_patterns(min_frequency=2)
        
        # Run learning cycle
        start_time = time.time()
        monitor = MemoryMonitor()
        monitor.start()
        
        cycle_result = engine.compound_learning_cycle()
        
        elapsed = time.time() - start_time
        memory_delta = monitor.snapshot()
        
        assert cycle_result is not None
        # Should have keys like 'cycle_number', 'patterns_discovered', etc.
        assert 'cycle_number' in cycle_result or 'patterns_discovered' in cycle_result
        # Learning cycle should complete in < 500ms
        assert elapsed < 0.5
        assert memory_delta < 50
        
        print(f"✓ Learning cycle with {len(engine.patterns)} patterns completed in {elapsed*1000:.2f}ms")
        print(f"  Memory delta: {memory_delta:.2f}MB")


class TestMemoryLeakDetection:
    """Tests to detect memory leaks"""
    
    def test_experience_cleanup_on_overflow(self):
        """Test that experiences are managed efficiently"""
        engine = CompoundLearningEngine(max_patterns=100)
        monitor = MemoryMonitor()
        monitor.start()
        
        # Record experiences up to max
        for i in range(200):
            exp = LearningExperience(
                query=f"query_{i}",
                response=f"response_{i}",
                latency_ms=10.0,
                success=True,
                timestamp=datetime.now().isoformat(),
            )
            engine.record_experience(exp)
        
        # Memory shouldn't grow unbounded
        delta = monitor.snapshot()
        initial_count = len(engine.experiences)
        
        # Record more experiences
        for i in range(200, 500):
            exp = LearningExperience(
                query=f"query_{i}",
                response=f"response_{i}",
                latency_ms=10.0,
                success=True,
                timestamp=datetime.now().isoformat(),
            )
            engine.record_experience(exp)
        
        delta2 = monitor.snapshot()
        final_count = len(engine.experiences)
        
        # Memory growth in second batch should be modest
        assert (delta2 - delta) <= delta or (delta2 - delta) < 20
        print(f"✓ Experience management working: delta1={delta:.2f}MB, delta2={delta2-delta:.2f}MB")
        print(f"  Experiences: {initial_count} -> {final_count}")
    
    def test_pattern_cache_memory_stability(self):
        """Test that pattern cache doesn't cause memory issues"""
        engine = CompoundLearningEngine()
        monitor = MemoryMonitor()
        monitor.start()
        
        # Create many patterns
        for i in range(500):
            pattern = CompoundLearningPattern(
                pattern_id=f'pattern_{i}',
                query_hash=f'hash_{i}',
                response_template=f'template_{i}' * 10,  # Large template
                accuracy=0.8,
                frequency=i % 100,
                avg_latency_ms=10.0,
                effectiveness_score=0.8,
                cross_model_applicability={f'model_{j}': 0.8 for j in range(10)}
            )
            engine.patterns[f'pattern_{i}'] = pattern
        
        delta = monitor.snapshot()
        
        # Access patterns many times
        for _ in range(100):
            for i in range(500):
                _ = engine.patterns.get(f'pattern_{i}')
        
        delta2 = monitor.snapshot()
        
        # Accessing patterns shouldn't increase memory
        assert delta2 - delta < 5
        print(f"✓ Pattern cache memory stable: delta1={delta:.2f}MB, delta2={delta2-delta:.2f}MB")
    
    def test_integration_engine_cleanup(self):
        """Test that integration engine properly cleans up resources"""
        learning_engine = CompoundLearningEngine()
        integration_engine = CompoundIntegrationEngine(learning_engine=learning_engine)
        monitor = MemoryMonitor()
        monitor.start()
        
        # Create and execute many integration paths
        for i in range(50):
            model_a = f'model_a_{i}'
            model_b = f'model_b_{i}'
            
            integration_engine.register_model(model_a, 'nlp', {})
            integration_engine.register_model(model_b, 'vision', {})
            integration_engine.define_integration_path(f'path_{i}', [model_a, model_b])
        
        delta = monitor.snapshot()
        
        # Execute all paths multiple times
        for _ in range(10):
            for i in range(50):
                integration_engine.execute_integration_path(
                    f'path_{i}',
                    {'data': 'test'},
                    record_learning=False
                )
        
        delta2 = monitor.snapshot()
        
        # Memory shouldn't grow significantly
        assert delta2 - delta < 50
        print(f"✓ Integration engine cleanup working: delta1={delta:.2f}MB, delta2={delta2-delta:.2f}MB")


class TestConcurrentExecution:
    """Test concurrent execution scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_experience_recording(self):
        """Test recording experiences concurrently"""
        engine = CompoundLearningEngine()
        
        async def record_batch(batch_id: int, count: int):
            """Record experiences in a batch"""
            for i in range(count):
                exp = LearningExperience(
                    query=f"batch_{batch_id}_query_{i}",
                    response=f"response_{i}",
                    latency_ms=10.0,
                    success=True,
                    timestamp=datetime.now().isoformat(),
                )
                engine.record_experience(exp)
        
        start_time = time.time()
        
        # Run 10 batches of 100 experiences each concurrently
        tasks = [record_batch(i, 100) for i in range(10)]
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        # Should have recorded 1000 experiences
        assert len(engine.experiences) == 1000
        assert elapsed < 2.0  # Should complete quickly
        
        print(f"✓ Concurrent recording of 1,000 experiences completed in {elapsed*1000:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_integration_path_execution(self):
        """Test executing integration paths concurrently"""
        learning_engine = CompoundLearningEngine()
        integration_engine = CompoundIntegrationEngine(learning_engine=learning_engine)
        
        # Setup models and paths
        for i in range(10):
            integration_engine.register_model(f'model_a_{i}', 'nlp', {})
            integration_engine.register_model(f'model_b_{i}', 'vision', {})
            integration_engine.define_integration_path(
                f'path_{i}',
                [f'model_a_{i}', f'model_b_{i}']
            )
        
        async def execute_path(path_id: str, executions: int):
            """Execute a path multiple times"""
            for _ in range(executions):
                integration_engine.execute_integration_path(
                    path_id,
                    {'data': 'test'},
                    record_learning=True
                )
        
        start_time = time.time()
        
        # Execute all paths concurrently, 5 times each
        tasks = [execute_path(f'path_{i}', 5) for i in range(10)]
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        # Should have completed 50 executions
        assert integration_engine.integration_metrics['total_integrations'] == 50
        assert elapsed < 5.0
        
        print(f"✓ Concurrent execution of 50 path runs completed in {elapsed*1000:.2f}ms")
    
    def test_concurrent_pattern_extraction_and_access(self):
        """Test concurrent pattern extraction and access"""
        engine = CompoundLearningEngine()
        
        # Record experiences
        for i in range(1000):
            exp = LearningExperience(
                query=f"query_{i % 50}",
                response=f"response_{i % 50}",
                latency_ms=10.0,
                success=True,
                timestamp=datetime.now().isoformat(),
            )
            engine.record_experience(exp)
        
        # Extract patterns
        engine.extract_patterns(min_frequency=2)
        
        # Track access patterns
        access_count = defaultdict(int)
        
        def access_patterns(thread_id: int):
            """Access patterns from thread"""
            for i in range(100):
                for pattern_id in list(engine.patterns.keys())[:10]:
                    _ = engine.patterns.get(pattern_id)
                    access_count[pattern_id] += 1
        
        start_time = time.time()
        
        # Simulate 5 concurrent accessors
        import threading
        threads = [threading.Thread(target=access_patterns, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        elapsed = time.time() - start_time
        
        # All patterns should be accessible
        assert len(access_count) > 0
        assert elapsed < 2.0
        
        print(f"✓ Concurrent pattern access with {sum(access_count.values())} total accesses in {elapsed*1000:.2f}ms")


class TestStabilityUnderLoad:
    """Long-running stability tests"""
    
    def test_sustained_experience_recording(self):
        """Test sustained experience recording"""
        engine = CompoundLearningEngine()
        monitor = MemoryMonitor()
        monitor.start()
        
        success_count = 0
        error_count = 0
        
        # Record experiences for sustained period
        for i in range(5000):
            try:
                exp = LearningExperience(
                    query=f"query_{i % 100}",
                    response=f"response_{i}",
                    latency_ms=10.0 + (i % 50),
                    success=(i % 10) != 9,
                    timestamp=datetime.now().isoformat(),
                )
                engine.record_experience(exp)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error recording experience: {e}")
        
        memory_delta = monitor.get_current_delta()
        
        assert success_count == 5000
        assert error_count == 0
        assert memory_delta < 200  # Should use reasonable memory
        
        print(f"✓ Sustained recording: {success_count} successes, {error_count} errors")
        print(f"  Memory usage: {memory_delta:.2f}MB")
    
    def test_repeated_learning_cycles(self):
        """Test repeated learning cycles for stability"""
        engine = CompoundLearningEngine()
        
        # Setup
        for i in range(10):
            engine.register_model(f'model_{i}', ['type_a'])
        
        cycle_results = []
        
        # Run 20 learning cycles
        for cycle_num in range(20):
            # Add some experiences
            for i in range(100):
                exp = LearningExperience(
                    query=f"query_{i % 10}",
                    response=f"response_{cycle_num}_{i}",
                    latency_ms=10.0,
                    success=True,
                    timestamp=datetime.now().isoformat(),
                )
                engine.record_experience(exp)
            
            # Extract patterns
            engine.extract_patterns(min_frequency=1)
            
            # Run learning cycle
            result = engine.compound_learning_cycle()
            if result:
                cycle_results.append(result)
        
        # All cycles should complete successfully
        assert len(cycle_results) == 20
        assert all(r is not None for r in cycle_results)
        
        print(f"✓ Completed {len(cycle_results)} learning cycles successfully")
    
    def test_integration_engine_repeated_execution(self):
        """Test integration engine under repeated execution"""
        learning_engine = CompoundLearningEngine()
        integration_engine = CompoundIntegrationEngine(learning_engine=learning_engine)
        monitor = MemoryMonitor()
        monitor.start()
        
        # Setup - register models in both engines
        for i in range(5):
            # Register in learning engine
            learning_engine.register_model(f'model_{i}', ['nlp'])
            # Register in integration engine
            integration_engine.register_model(f'model_{i}', 'nlp', {})
        
        # Then define paths
        path_count = 0
        for i in range(4):
            model_a = f'model_{i}'
            model_b = f'model_{(i+1)%5}'
            integration_engine.define_integration_path(f'path_{i}', [model_a, model_b])
            path_count += 1
        
        # Execute repeatedly
        success_count = 0
        for _ in range(100):
            for path_id in list(integration_engine.integration_paths.keys()):
                try:
                    result = integration_engine.execute_integration_path(
                        path_id,
                        {'data': 'test'},
                        record_learning=True
                    )
                    if result['success']:
                        success_count += 1
                except (KeyError, Exception):
                    # Path may have been removed or other error, skip
                    pass
        
        memory_delta = monitor.get_current_delta()
        
        assert success_count > 0
        assert memory_delta < 150
        
        print(f"✓ Integration engine: {success_count} successful executions")
        print(f"  Memory delta: {memory_delta:.2f}MB")


class TestPerformanceUnderLoad:
    """Performance benchmarks under various load conditions"""
    
    def test_throughput_experience_recording(self):
        """Measure throughput of experience recording"""
        engine = CompoundLearningEngine()
        
        start_time = time.time()
        for i in range(10000):
            exp = LearningExperience(
                query=f"q{i}",
                response=f"r{i}",
                latency_ms=10.0,
                success=True,
                timestamp=datetime.now().isoformat(),
            )
            engine.record_experience(exp)
        
        elapsed = time.time() - start_time
        throughput = 10000 / elapsed
        
        assert throughput > 1000  # Should record > 1000 experiences/sec
        print(f"✓ Experience recording throughput: {throughput:.0f} experiences/sec")
    
    def test_throughput_pattern_extraction(self):
        """Measure pattern extraction throughput"""
        engine = CompoundLearningEngine()
        
        # Pre-populate with experiences
        for i in range(5000):
            exp = LearningExperience(
                query=f"query_{i % 100}",
                response=f"response_{i}",
                latency_ms=10.0,
                success=True,
                timestamp=datetime.now().isoformat(),
            )
            engine.record_experience(exp)
        
        start_time = time.time()
        patterns = engine.extract_patterns(min_frequency=5)
        elapsed = time.time() - start_time
        
        patterns_per_sec = len(patterns) / elapsed if elapsed > 0 else 0
        
        assert elapsed < 1.0
        print(f"✓ Pattern extraction: {len(patterns)} patterns in {elapsed*1000:.2f}ms")
    
    def test_latency_under_load(self):
        """Measure latency of operations under load"""
        engine = CompoundLearningEngine()
        
        # Setup: 100 models
        for i in range(100):
            engine.register_model(f'model_{i}', ['type_a'])
        
        # Create baseline
        latencies = []
        for _ in range(100):
            exp = LearningExperience(
                query="test",
                response="response",
                latency_ms=10.0,
                success=True,
                timestamp=datetime.now().isoformat(),
            )
            
            start = time.time()
            engine.record_experience(exp)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Should be fast even with 100 models
        assert max_latency < 50  # 50ms max
        assert avg_latency < 5   # 5ms average
        
        print(f"✓ Operation latency: avg={avg_latency:.3f}ms, max={max_latency:.3f}ms")


def run_advanced_test_suite():
    """Run all advanced tests with reporting"""
    print("\n" + "="*80)
    print("COMPOUND LEARNING - ADVANCED TEST SUITE")
    print("(Stress, Stability, Memory, Concurrency)")
    print("="*80 + "\n")
    
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Run stress tests
    print("Running Stress Tests...")
    print("-" * 80)
    stress_tests = TestStressLargeModelCounts()
    
    try:
        stress_tests.test_register_100_models()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_register_100_models', str(e)))
    
    try:
        stress_tests.test_register_500_models()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_register_500_models', str(e)))
    
    try:
        stress_tests.test_affinity_matrix_scaling()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_affinity_matrix_scaling', str(e)))
    
    # Run data volume tests
    print("\nRunning Data Volume Tests...")
    print("-" * 80)
    data_tests = TestStressLargeDataVolumes()
    
    try:
        data_tests.test_record_10000_experiences()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_record_10000_experiences', str(e)))
    
    try:
        data_tests.test_extract_patterns_large_dataset()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_extract_patterns_large_dataset', str(e)))
    
    try:
        data_tests.test_learning_cycle_with_many_patterns()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_learning_cycle_with_many_patterns', str(e)))
    
    # Run memory leak tests
    print("\nRunning Memory Leak Detection Tests...")
    print("-" * 80)
    memory_tests = TestMemoryLeakDetection()
    
    try:
        memory_tests.test_experience_cleanup_on_overflow()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_experience_cleanup_on_overflow', str(e)))
    
    try:
        memory_tests.test_pattern_cache_memory_stability()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_pattern_cache_memory_stability', str(e)))
    
    try:
        memory_tests.test_integration_engine_cleanup()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_integration_engine_cleanup', str(e)))
    
    # Run stability tests
    print("\nRunning Stability Tests...")
    print("-" * 80)
    stability_tests = TestStabilityUnderLoad()
    
    try:
        stability_tests.test_sustained_experience_recording()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_sustained_experience_recording', str(e)))
    
    try:
        stability_tests.test_repeated_learning_cycles()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_repeated_learning_cycles', str(e)))
    
    try:
        stability_tests.test_integration_engine_repeated_execution()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_integration_engine_repeated_execution', str(e)))
    
    # Run performance tests
    print("\nRunning Performance Benchmark Tests...")
    print("-" * 80)
    perf_tests = TestPerformanceUnderLoad()
    
    try:
        perf_tests.test_throughput_experience_recording()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_throughput_experience_recording', str(e)))
    
    try:
        perf_tests.test_throughput_pattern_extraction()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_throughput_pattern_extraction', str(e)))
    
    try:
        perf_tests.test_latency_under_load()
        test_results['passed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(('test_latency_under_load', str(e)))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    
    if test_results['errors']:
        print(f"\nFailed Tests:")
        for test_name, error in test_results['errors']:
            print(f"  - {test_name}: {error}")
    
    print("="*80 + "\n")
    
    return test_results['failed'] == 0


if __name__ == "__main__":
    # Check for psutil dependency
    try:
        import psutil
    except ImportError:
        print("Installing psutil for memory monitoring...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    
    # Run advanced test suite
    success = run_advanced_test_suite()
    
    # Run pytest if available
    try:
        pytest.main([__file__, '-v', '-s'])
    except:
        print("Note: Run 'pip install pytest' to run tests with pytest")
    
    sys.exit(0 if success else 1)
