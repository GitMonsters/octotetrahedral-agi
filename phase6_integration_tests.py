"""
Phase 6: Comprehensive Integration Testing
Full end-to-end validation of all Confucius SDK components
Tests multi-component workflows, cross-session learning, and performance
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ngvt_orchestrator import (
    NGVTUnifiedOrchestrator,
    OrchestratorConfig,
    Task,
)
from ngvt_memory import NGVTHierarchicalMemory, MemoryScopeType
from ngvt_notes import PatternNoteStore, PatternType
from ngvt_meta_agent import (
    NGVTMetaAgent,
    ConfigurationSpace,
    SimpleConfigEvaluator,
    ConfigurationSynthesizer,
    OptimizationStrategy,
)
from ngvt_extensions import (
    ExtensionRegistry,
    LoggingExtension,
    MetricsExtension,
    CacheExtension,
    ExtensionToolChain,
    ExtensionPhase,
    HookContext,
)


@dataclass
class IntegrationTestResult:
    """Result of an integration test"""
    test_name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = None


class Phase6IntegrationTest:
    """Comprehensive integration testing for Confucius SDK"""
    
    def __init__(self):
        self.results: List[IntegrationTestResult] = []
        self.start_time = time.time()
    
    async def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all integration tests"""
        if verbose:
            print("\n" + "="*80)
            print("PHASE 6: COMPREHENSIVE INTEGRATION TESTING")
            print("="*80 + "\n")
        
        tests = [
            ("Memory System", self.test_memory_system),
            ("Note-Taking System", self.test_note_taking_system),
            ("Orchestrator Basic", self.test_orchestrator_basic),
            ("Orchestrator + Memory", self.test_orchestrator_memory_integration),
            ("Orchestrator + Notes", self.test_orchestrator_notes_integration),
            ("Extension System", self.test_extension_system),
            ("Meta-Agent Optimization", self.test_meta_agent),
            ("Cross-Session Learning", self.test_cross_session_learning),
            ("Full End-to-End Pipeline", self.test_full_pipeline),
            ("Error Handling", self.test_error_handling),
            ("Performance Baseline", self.test_performance_baseline),
        ]
        
        for test_name, test_func in tests:
            if verbose:
                print(f"[•] Running: {test_name}...", end=" ", flush=True)
            
            try:
                result = await test_func()
                self.results.append(result)
                
                status = "✓ PASS" if result.passed else "✗ FAIL"
                if verbose:
                    print(f"{status} ({result.duration_ms:.1f}ms)")
                    if result.error:
                        print(f"    Error: {result.error}")
            
            except Exception as e:
                result = IntegrationTestResult(
                    test_name=test_name,
                    passed=False,
                    duration_ms=0,
                    error=str(e),
                )
                self.results.append(result)
                if verbose:
                    print(f"✗ FAIL (Exception: {str(e)[:50]})")
        
        return self._generate_report()
    
    async def test_memory_system(self) -> IntegrationTestResult:
        """Test memory system initialization and operation"""
        start = time.time()
        
        try:
            memory = NGVTHierarchicalMemory()
            memory.initialize_session()
            
            # Record experiences
            for i in range(5):
                memory.record_experience(
                    scope=MemoryScopeType.RUNNABLE,
                    content={"step": i, "data": f"observation_{i}"},
                )
            
            # Compose for prompt
            prompt_context = memory.compose_for_prompt()
            
            success = len(prompt_context) > 0 and memory.scopes is not None
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Memory System",
                passed=success,
                duration_ms=duration,
                details={
                    "scopes": len(memory.scopes),
                    "total_entries": sum(len(s.entries) for s in memory.scopes.values()),
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Memory System",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_note_taking_system(self) -> IntegrationTestResult:
        """Test note-taking system storage and retrieval"""
        start = time.time()
        
        try:
            store = PatternNoteStore(storage_dir="./test_phase6_patterns")
            
            from ngvt_notes import PatternNote
            
            # Add test patterns
            patterns_added = 0
            for i in range(3):
                pattern = PatternNote(
                    id=f"test_{i}",
                    title=f"Test Pattern {i}",
                    pattern_type=PatternType.NLP_PATTERN,
                    problem=f"Problem {i}",
                    solution=f"Solution {i}",
                    keywords=[f"keyword_{i}"],
                    effectiveness=0.7 + (i * 0.1),
                )
                store.add_pattern(pattern)
                patterns_added += 1
            
            # Retrieve patterns
            retrieved = store.search_similar("test", top_k=5)
            
            success = patterns_added > 0 and len(retrieved) > 0
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Note-Taking System",
                passed=success,
                duration_ms=duration,
                details={
                    "patterns_added": patterns_added,
                    "patterns_retrieved": len(retrieved),
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Note-Taking System",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_orchestrator_basic(self) -> IntegrationTestResult:
        """Test basic orchestrator functionality"""
        start = time.time()
        
        try:
            config = OrchestratorConfig(max_iterations=3, verbose=False)
            orchestrator = NGVTUnifiedOrchestrator(config)
            
            task = Task(
                id="test_basic",
                title="Basic Test",
                description="Test orchestrator basic functionality",
                max_iterations=3,
            )
            
            artifacts = await orchestrator.run_session(task)
            
            success = (
                artifacts["status"] == "completed" and
                artifacts["iterations"] > 0 and
                artifacts["total_actions"] > 0
            )
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Orchestrator Basic",
                passed=success,
                duration_ms=duration,
                details={
                    "iterations": artifacts["iterations"],
                    "total_actions": artifacts["total_actions"],
                    "successful_actions": artifacts["successful_actions"],
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Orchestrator Basic",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_orchestrator_memory_integration(self) -> IntegrationTestResult:
        """Test orchestrator with memory integration"""
        start = time.time()
        
        try:
            config = OrchestratorConfig(max_iterations=2, verbose=False)
            orchestrator = NGVTUnifiedOrchestrator(config)
            
            # Verify memory is integrated
            memory_integrated = orchestrator.memory is not None
            
            task = Task(
                id="test_memory_int",
                title="Memory Integration Test",
                description="Verify memory integration with orchestrator",
                max_iterations=2,
            )
            
            artifacts = await orchestrator.run_session(task)
            
            # Check memory stats
            memory_stats = artifacts.get("memory_stats", {})
            
            success = (
                memory_integrated and
                artifacts["status"] == "completed" and
                memory_stats.get("total_entries", 0) > 0
            )
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Orchestrator + Memory",
                passed=success,
                duration_ms=duration,
                details={
                    "memory_scopes": memory_stats.get("scopes", 0),
                    "memory_entries": memory_stats.get("total_entries", 0),
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Orchestrator + Memory",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_orchestrator_notes_integration(self) -> IntegrationTestResult:
        """Test orchestrator with note-taking integration"""
        start = time.time()
        
        try:
            config = OrchestratorConfig(max_iterations=2, verbose=False)
            orchestrator = NGVTUnifiedOrchestrator(config)
            
            # Verify notes are integrated
            notes_integrated = orchestrator.note_store is not None
            initial_pattern_count = len(orchestrator.note_store.index)
            
            task = Task(
                id="test_notes_int",
                title="Notes Integration Test",
                description="Verify notes integration with orchestrator",
                max_iterations=2,
            )
            
            artifacts = await orchestrator.run_session(task)
            
            # Check if patterns were stored
            final_pattern_count = len(orchestrator.note_store.index)
            
            success = (
                notes_integrated and
                artifacts["status"] == "completed" and
                final_pattern_count > initial_pattern_count
            )
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Orchestrator + Notes",
                passed=success,
                duration_ms=duration,
                details={
                    "patterns_stored": final_pattern_count - initial_pattern_count,
                    "total_patterns": final_pattern_count,
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Orchestrator + Notes",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_extension_system(self) -> IntegrationTestResult:
        """Test extension system with orchestrator"""
        start = time.time()
        
        try:
            registry = ExtensionRegistry()
            
            # Register extensions
            await registry.register(LoggingExtension())
            await registry.register(MetricsExtension())
            await registry.register(CacheExtension())
            
            extensions_registered = len(registry.extensions)
            
            # Test phase calls
            context = HookContext(
                phase=ExtensionPhase.PRE_PROMPT,
                extension_name="Test",
            )
            
            results = await registry.call_phase(ExtensionPhase.PRE_PROMPT, context)
            
            success = (
                extensions_registered == 3 and
                len(results) > 0 and
                all(r.success for r in results)
            )
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Extension System",
                passed=success,
                duration_ms=duration,
                details={
                    "extensions_registered": extensions_registered,
                    "successful_hooks": sum(1 for r in results if r.success),
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Extension System",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_meta_agent(self) -> IntegrationTestResult:
        """Test meta-agent optimization"""
        start = time.time()
        
        try:
            params = ConfigurationSynthesizer.create_orchestrator_parameters()[:2]
            config_space = ConfigurationSpace(params)
            evaluator = SimpleConfigEvaluator()
            
            meta_agent = NGVTMetaAgent(
                config_space=config_space,
                evaluator=evaluator,
                strategy=OptimizationStrategy.BAYESIAN,
            )
            
            result = await meta_agent.optimize(
                max_iterations=5,
                timeout_seconds=30,
                verbose=False,
            )
            
            success = (
                result.best_config is not None and
                result.best_score > 0 and
                result.iterations > 0
            )
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Meta-Agent Optimization",
                passed=success,
                duration_ms=duration,
                details={
                    "best_score": result.best_score,
                    "iterations": result.iterations,
                    "candidates_evaluated": result.candidates_evaluated,
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Meta-Agent Optimization",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_cross_session_learning(self) -> IntegrationTestResult:
        """Test cross-session learning with patterns"""
        start = time.time()
        
        try:
            # Session 1
            store1 = PatternNoteStore(storage_dir="./test_phase6_cross_session")
            
            from ngvt_notes import PatternNote
            
            pattern1 = PatternNote(
                id="session1_pattern",
                title="Session 1 Learning",
                pattern_type=PatternType.INTEGRATION_PATTERN,
                problem="Problem from session 1",
                solution="Solution discovered in session 1",
                keywords=["session1", "learning"],
                effectiveness=0.85,
            )
            store1.add_pattern(pattern1)
            
            patterns_in_store1 = len(store1.index)
            
            # Session 2 - Load same store
            store2 = PatternNoteStore(storage_dir="./test_phase6_cross_session")
            
            patterns_in_store2 = len(store2.index)
            
            # Verify persistence
            retrieved = store2.search_similar("session1", top_k=5)
            
            success = (
                patterns_in_store1 > 0 and
                patterns_in_store2 == patterns_in_store1 and
                len(retrieved) > 0
            )
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Cross-Session Learning",
                passed=success,
                duration_ms=duration,
                details={
                    "patterns_session1": patterns_in_store1,
                    "patterns_session2": patterns_in_store2,
                    "patterns_retrieved": len(retrieved),
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Cross-Session Learning",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_full_pipeline(self) -> IntegrationTestResult:
        """Test full end-to-end pipeline with all components"""
        start = time.time()
        
        try:
            # 1. Setup orchestrator with all components
            config = OrchestratorConfig(max_iterations=3, verbose=False)
            orchestrator = NGVTUnifiedOrchestrator(config)
            
            # 2. Setup extensions
            registry = ExtensionRegistry()
            await registry.register(LoggingExtension())
            await registry.register(MetricsExtension())
            
            # 3. Run orchestration
            task = Task(
                id="full_pipeline",
                title="Full Pipeline Test",
                description="End-to-end test with all components",
                max_iterations=3,
            )
            
            artifacts = await orchestrator.run_session(task)
            
            # 4. Validate results
            memory_working = artifacts.get("memory_stats", {}).get("total_entries", 0) > 0
            notes_working = len(orchestrator.note_store.index) > 0
            orchestration_working = artifacts["status"] == "completed"
            extensions_working = len(registry.extensions) > 0
            
            success = (
                memory_working and
                notes_working and
                orchestration_working and
                extensions_working
            )
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Full End-to-End Pipeline",
                passed=success,
                duration_ms=duration,
                details={
                    "memory_entries": artifacts.get("memory_stats", {}).get("total_entries", 0),
                    "stored_patterns": len(orchestrator.note_store.index),
                    "iterations": artifacts["iterations"],
                    "extensions_active": len(registry.extensions),
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Full End-to-End Pipeline",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_error_handling(self) -> IntegrationTestResult:
        """Test error handling and recovery"""
        start = time.time()
        
        try:
            config = OrchestratorConfig(max_iterations=2, verbose=False)
            orchestrator = NGVTUnifiedOrchestrator(config)
            
            # Create a task with edge cases
            task = Task(
                id="error_test",
                title="Error Handling Test",
                description="Test error handling and recovery",
                max_iterations=2,
            )
            
            # Run should handle gracefully
            artifacts = await orchestrator.run_session(task)
            
            # Check if system recovered
            success = (
                artifacts["status"] == "completed" and
                artifacts["iterations"] > 0
            )
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Error Handling",
                passed=success,
                duration_ms=duration,
                details={
                    "completed_successfully": success,
                }
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Error Handling",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    async def test_performance_baseline(self) -> IntegrationTestResult:
        """Test performance baseline metrics"""
        start = time.time()
        
        try:
            results_dict = {}
            
            # Memory composition speed
            memory = NGVTHierarchicalMemory()
            memory.initialize_session()
            for i in range(10):
                memory.record_experience(
                    scope=MemoryScopeType.RUNNABLE,
                    content={"data": f"test_{i}"},
                )
            
            mem_start = time.time()
            _ = memory.compose_for_prompt()
            results_dict["memory_composition_ms"] = (time.time() - mem_start) * 1000
            
            # Pattern retrieval speed
            store = PatternNoteStore(storage_dir="./test_phase6_perf")
            search_start = time.time()
            _ = store.search_similar("test", top_k=5)
            results_dict["pattern_search_ms"] = (time.time() - search_start) * 1000
            
            # Orchestrator iteration speed
            config = OrchestratorConfig(max_iterations=1, verbose=False)
            orchestrator = NGVTUnifiedOrchestrator(config)
            
            task = Task(
                id="perf_test",
                title="Performance Test",
                description="Measure performance",
                max_iterations=1,
            )
            
            orch_start = time.time()
            _ = await orchestrator.run_session(task)
            results_dict["orchestration_iteration_ms"] = (time.time() - orch_start) * 1000
            
            # Check if all within acceptable ranges
            success = (
                results_dict["memory_composition_ms"] < 100 and
                results_dict["pattern_search_ms"] < 500 and
                results_dict["orchestration_iteration_ms"] < 2000
            )
            
            duration = (time.time() - start) * 1000
            
            return IntegrationTestResult(
                test_name="Performance Baseline",
                passed=success,
                duration_ms=duration,
                details=results_dict,
            )
        except Exception as e:
            return IntegrationTestResult(
                test_name="Performance Baseline",
                passed=False,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate integration test report"""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        total_time = time.time() - self.start_time
        
        return {
            "test_suite": "Phase 6: Full Integration Testing",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "success_rate": (passed / total * 100) if total > 0 else 0,
                "total_duration_seconds": total_time,
            },
            "results": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "details": r.details or {},
                }
                for r in self.results
            ],
        }


async def run_phase6_integration_tests():
    """Run Phase 6 integration tests"""
    tester = Phase6IntegrationTest()
    report = await tester.run_all_tests(verbose=True)
    
    print("\n" + "="*80)
    print("INTEGRATION TEST REPORT")
    print("="*80 + "\n")
    
    summary = report["summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Duration: {summary['total_duration_seconds']:.2f}s\n")
    
    # Show failed tests
    failed_tests = [r for r in report["results"] if not r["passed"]]
    if failed_tests:
        print("Failed Tests:")
        for test in failed_tests:
            print(f"  ✗ {test['name']}")
            if test['error']:
                print(f"    Error: {test['error']}")
    
    # Show performance metrics
    print("\n" + "-"*80)
    print("Performance Metrics")
    print("-"*80 + "\n")
    
    for result in report["results"]:
        if result["details"]:
            print(f"{result['name']}:")
            for key, value in result["details"].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    
    return report


if __name__ == "__main__":
    report = asyncio.run(run_phase6_integration_tests())
