"""
Comprehensive test suite for NGVTUnifiedOrchestrator
Tests orchestration loop, memory integration, pattern system, and extension routing
"""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from ngvt_orchestrator import (
    NGVTUnifiedOrchestrator,
    OrchestratorConfig,
    Task,
    Action,
    ActionType,
    Observation,
    InferenceExtension,
    PatternExtension,
    EvaluationExtension,
    ExtensionRegistry,
)
from ngvt_memory import NGVTHierarchicalMemory, MemoryScopeType
from ngvt_notes import PatternNoteStore, PatternType


class TestAction:
    """Tests for Action dataclass"""
    
    def test_action_creation(self):
        """Test creating an action"""
        action = Action(
            type=ActionType.INFERENCE,
            params={"prompt": "test"},
            reasoning="test reasoning"
        )
        assert action.type == ActionType.INFERENCE
        assert action.params["prompt"] == "test"
        assert action.reasoning == "test reasoning"
    
    def test_action_timestamp(self):
        """Test action has timestamp"""
        action = Action(
            type=ActionType.INFERENCE,
            params={},
            reasoning=""
        )
        assert action.timestamp is not None
        assert isinstance(action.timestamp, str)


class TestObservation:
    """Tests for Observation dataclass"""
    
    def test_observation_success(self):
        """Test creating successful observation"""
        obs = Observation(
            action_type=ActionType.INFERENCE,
            success=True,
            result={"output": "test"},
            latency_ms=100.0
        )
        assert obs.success is True
        assert obs.result["output"] == "test"
        assert obs.latency_ms == 100.0
    
    def test_observation_failure(self):
        """Test creating failed observation"""
        obs = Observation(
            action_type=ActionType.INFERENCE,
            success=False,
            result=None,
            error="Connection timeout"
        )
        assert obs.success is False
        assert obs.error == "Connection timeout"


class TestInferenceExtension:
    """Tests for InferenceExtension"""
    
    @pytest.mark.asyncio
    async def test_can_handle_inference_action(self):
        """Test inference extension recognizes inference actions"""
        config = OrchestratorConfig()
        ext = InferenceExtension(config)
        
        action = Action(
            type=ActionType.INFERENCE,
            params={},
            reasoning=""
        )
        
        can_handle = await ext.can_handle(action)
        assert can_handle is True
    
    @pytest.mark.asyncio
    async def test_cannot_handle_other_actions(self):
        """Test inference extension rejects non-inference actions"""
        config = OrchestratorConfig()
        ext = InferenceExtension(config)
        
        action = Action(
            type=ActionType.RETRIEVE_PATTERN,
            params={},
            reasoning=""
        )
        
        can_handle = await ext.can_handle(action)
        assert can_handle is False
    
    @pytest.mark.asyncio
    async def test_execute_inference_failure(self):
        """Test inference extension handles connection failure gracefully"""
        config = OrchestratorConfig(extension_timeout_ms=1000)
        ext = InferenceExtension(config, server_port=9999)  # Likely unused port
        
        action = Action(
            type=ActionType.INFERENCE,
            params={"prompt": "test"},
            reasoning=""
        )
        
        obs = await ext.execute(action)
        assert obs.success is False
        assert obs.error is not None
        assert obs.latency_ms > 0
    
    def test_extension_stats(self):
        """Test extension statistics tracking"""
        config = OrchestratorConfig()
        ext = InferenceExtension(config)
        
        stats = ext.get_stats()
        assert stats["name"] == "InferenceExtension"
        assert stats["execution_count"] == 0


class TestPatternExtension:
    """Tests for PatternExtension"""
    
    @pytest.mark.asyncio
    async def test_can_handle_retrieve_pattern(self):
        """Test pattern extension recognizes retrieve pattern actions"""
        config = OrchestratorConfig()
        note_store = PatternNoteStore(storage_dir="./test_patterns")
        ext = PatternExtension(config, note_store)
        
        action = Action(
            type=ActionType.RETRIEVE_PATTERN,
            params={},
            reasoning=""
        )
        
        can_handle = await ext.can_handle(action)
        assert can_handle is True
    
    @pytest.mark.asyncio
    async def test_can_handle_store_pattern(self):
        """Test pattern extension recognizes store pattern actions"""
        config = OrchestratorConfig()
        note_store = PatternNoteStore(storage_dir="./test_patterns")
        ext = PatternExtension(config, note_store)
        
        action = Action(
            type=ActionType.STORE_PATTERN,
            params={},
            reasoning=""
        )
        
        can_handle = await ext.can_handle(action)
        assert can_handle is True
    
    @pytest.mark.asyncio
    async def test_execute_retrieve_pattern(self):
        """Test retrieving patterns"""
        config = OrchestratorConfig()
        note_store = PatternNoteStore(storage_dir="./test_patterns")
        
        # Add test pattern
        note_store.add_pattern(
            id="test_1",
            title="Test Pattern",
            pattern_type=PatternType.NLP_PATTERN,
            problem="Test problem",
            solution="Test solution",
            keywords=["test"],
            effectiveness=0.8
        )
        
        ext = PatternExtension(config, note_store)
        
        action = Action(
            type=ActionType.RETRIEVE_PATTERN,
            params={"query": "test", "limit": 5},
            reasoning=""
        )
        
        obs = await ext.execute(action)
        assert obs.success is True
        assert isinstance(obs.result, list)
        assert obs.latency_ms > 0


class TestEvaluationExtension:
    """Tests for EvaluationExtension"""
    
    @pytest.mark.asyncio
    async def test_can_handle_evaluate_action(self):
        """Test evaluation extension recognizes evaluate actions"""
        config = OrchestratorConfig()
        ext = EvaluationExtension(config)
        
        action = Action(
            type=ActionType.EVALUATE,
            params={},
            reasoning=""
        )
        
        can_handle = await ext.can_handle(action)
        assert can_handle is True
    
    @pytest.mark.asyncio
    async def test_evaluate_quality(self):
        """Test quality evaluation"""
        config = OrchestratorConfig()
        ext = EvaluationExtension(config)
        
        action = Action(
            type=ActionType.EVALUATE,
            params={"type": "quality", "data": "x" * 500},
            reasoning=""
        )
        
        obs = await ext.execute(action)
        assert obs.success is True
        assert "quality_score" in obs.result
        assert obs.result["passed"] is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_completeness(self):
        """Test completeness evaluation"""
        config = OrchestratorConfig()
        ext = EvaluationExtension(config)
        
        action = Action(
            type=ActionType.EVALUATE,
            params={
                "type": "completeness",
                "objectives": ["obj1", "obj2", "obj3"],
                "completed_count": 3
            },
            reasoning=""
        )
        
        obs = await ext.execute(action)
        assert obs.success is True
        assert obs.result["completeness"] == 1.0
        assert obs.result["passed"] is True


class TestExtensionRegistry:
    """Tests for ExtensionRegistry"""
    
    @pytest.mark.asyncio
    async def test_register_extension(self):
        """Test registering an extension"""
        registry = ExtensionRegistry()
        config = OrchestratorConfig()
        ext = InferenceExtension(config)
        
        registry.register(ext)
        assert "InferenceExtension" in registry.extensions
    
    @pytest.mark.asyncio
    async def test_route_to_extension(self):
        """Test routing action to correct extension"""
        registry = ExtensionRegistry()
        config = OrchestratorConfig()
        
        registry.register(InferenceExtension(config))
        registry.register(EvaluationExtension(config))
        
        action = Action(
            type=ActionType.INFERENCE,
            params={},
            reasoning=""
        )
        
        ext = await registry.route(action)
        assert ext is not None
        assert ext.name == "InferenceExtension"
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting statistics from all extensions"""
        registry = ExtensionRegistry()
        config = OrchestratorConfig()
        
        registry.register(InferenceExtension(config))
        registry.register(EvaluationExtension(config))
        
        stats = registry.get_stats()
        assert "InferenceExtension" in stats
        assert "EvaluationExtension" in stats


class TestNGVTUnifiedOrchestrator:
    """Tests for NGVTUnifiedOrchestrator"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly"""
        config = OrchestratorConfig(max_iterations=5)
        orchestrator = NGVTUnifiedOrchestrator(config)
        
        assert orchestrator.config.max_iterations == 5
        assert orchestrator.memory is not None
        assert orchestrator.note_store is not None
        assert len(orchestrator.extensions.extensions) == 3  # 3 default extensions
    
    def test_orchestrator_registers_default_extensions(self):
        """Test default extensions are registered"""
        orchestrator = NGVTUnifiedOrchestrator(OrchestratorConfig())
        
        names = list(orchestrator.extensions.extensions.keys())
        assert "InferenceExtension" in names
        assert "PatternExtension" in names
        assert "EvaluationExtension" in names
    
    def test_get_system_prompt(self):
        """Test system prompt generation"""
        orchestrator = NGVTUnifiedOrchestrator(OrchestratorConfig())
        prompt = orchestrator._get_system_prompt()
        
        assert "orchestration" in prompt.lower()
        assert "inference" in prompt.lower()
        assert "action" in prompt.lower()
    
    def test_get_actions_description(self):
        """Test actions description"""
        orchestrator = NGVTUnifiedOrchestrator(OrchestratorConfig())
        desc = orchestrator._get_actions_description()
        
        assert "inference" in desc.lower()
        assert "retrieve_pattern" in desc.lower()
        assert "evaluate" in desc.lower()
        assert "terminate" in desc.lower()
    
    def test_parse_actions(self):
        """Test action parsing logic"""
        orchestrator = NGVTUnifiedOrchestrator(OrchestratorConfig())
        prompt = "Test prompt requiring inference"
        
        actions = orchestrator._parse_actions(prompt)
        assert len(actions) > 0
        assert any(a.type == ActionType.INFERENCE for a in actions)
    
    def test_parse_actions_includes_termination(self):
        """Test actions include termination after iterations"""
        orchestrator = NGVTUnifiedOrchestrator(OrchestratorConfig())
        orchestrator.iteration_count = 3  # Force high iteration count
        
        actions = orchestrator._parse_actions("Test")
        assert any(a.type == ActionType.TERMINATE for a in actions)
    
    def test_compose_prompt_structure(self):
        """Test composed prompt contains expected components"""
        orchestrator = NGVTUnifiedOrchestrator(OrchestratorConfig())
        task = Task(
            id="test",
            title="Test Task",
            description="Test description"
        )
        orchestrator.session_context["task"] = task
        
        prompt = orchestrator._compose_prompt(task)
        
        assert "orchestration" in prompt.lower()
        assert "Test Task" in prompt
        assert "Test description" in prompt
        assert "Available Actions" in prompt
    
    def test_session_stats(self):
        """Test session statistics generation"""
        orchestrator = NGVTUnifiedOrchestrator(OrchestratorConfig())
        orchestrator.start_time = asyncio.get_event_loop().time()
        orchestrator.iteration_count = 3
        
        stats = orchestrator._get_session_stats()
        
        assert "Iteration: 3" in stats
        assert "Elapsed:" in stats
        assert "Total actions: 0" in stats
    
    def test_extract_artifacts(self):
        """Test artifact extraction"""
        orchestrator = NGVTUnifiedOrchestrator(OrchestratorConfig())
        orchestrator.session_context["task"] = Task(
            id="test",
            title="Test Task",
            description="Test"
        )
        orchestrator.iteration_count = 5
        orchestrator.actions_history = [
            Action(type=ActionType.INFERENCE, params={}, reasoning="")
        ]
        orchestrator.observations_history = [
            Observation(
                action_type=ActionType.INFERENCE,
                success=True,
                result={"output": "test"},
                latency_ms=100
            )
        ]
        
        artifacts = orchestrator._extract_artifacts(1.5)
        
        assert artifacts["task_id"] == "test"
        assert artifacts["task_title"] == "Test Task"
        assert artifacts["status"] == "completed"
        assert artifacts["iterations"] == 5
        assert artifacts["elapsed_seconds"] == 1.5
        assert artifacts["total_actions"] == 1
        assert artifacts["successful_actions"] == 1
    
    @pytest.mark.asyncio
    async def test_run_session_completes(self):
        """Test orchestrator session runs and completes"""
        config = OrchestratorConfig(max_iterations=3, verbose=False)
        orchestrator = NGVTUnifiedOrchestrator(config)
        
        task = Task(
            id="test",
            title="Test Task",
            description="Test task for orchestrator",
            max_iterations=3
        )
        
        artifacts = await orchestrator.run_session(task)
        
        assert artifacts["status"] == "completed"
        assert artifacts["task_id"] == "test"
        assert artifacts["iterations"] > 0
        assert artifacts["total_actions"] > 0
    
    @pytest.mark.asyncio
    async def test_run_session_respects_max_iterations(self):
        """Test session respects max iteration limit"""
        config = OrchestratorConfig(max_iterations=2, verbose=False)
        orchestrator = NGVTUnifiedOrchestrator(config)
        
        task = Task(
            id="test",
            title="Test",
            description="Test",
            max_iterations=2
        )
        
        artifacts = await orchestrator.run_session(task)
        
        assert artifacts["iterations"] <= 2
    
    @pytest.mark.asyncio
    async def test_run_session_memory_integration(self):
        """Test session integrates with memory system"""
        config = OrchestratorConfig(max_iterations=2, verbose=False)
        orchestrator = NGVTUnifiedOrchestrator(config)
        
        task = Task(
            id="test",
            title="Test",
            description="Test",
            max_iterations=2
        )
        
        artifacts = await orchestrator.run_session(task)
        
        # Check memory stats
        assert "memory_stats" in artifacts
        assert artifacts["memory_stats"] is not None
    
    @pytest.mark.asyncio
    async def test_run_session_stores_patterns(self):
        """Test session stores successful patterns"""
        config = OrchestratorConfig(max_iterations=2, verbose=False)
        orchestrator = NGVTUnifiedOrchestrator(config)
        
        initial_pattern_count = len(orchestrator.note_store.store)
        
        task = Task(
            id="test",
            title="Test",
            description="Test",
            max_iterations=2
        )
        
        artifacts = await orchestrator.run_session(task)
        
        # New pattern should be stored
        final_pattern_count = len(orchestrator.note_store.store)
        assert final_pattern_count > initial_pattern_count


class TestOrchestratorIntegration:
    """Integration tests for orchestrator with all components"""
    
    @pytest.mark.asyncio
    async def test_full_orchestration_workflow(self):
        """Test complete orchestration workflow"""
        config = OrchestratorConfig(
            max_iterations=4,
            verbose=False,
            temperature=0.7
        )
        orchestrator = NGVTUnifiedOrchestrator(config)
        
        task = Task(
            id="integration_test",
            title="Integration Test",
            description="Complete integration test of orchestration system",
            max_iterations=4
        )
        
        artifacts = await orchestrator.run_session(task)
        
        # Verify complete workflow
        assert artifacts["status"] == "completed"
        assert artifacts["total_actions"] > 0
        assert artifacts["successful_actions"] >= 0
        assert "extension_stats" in artifacts
        assert "memory_stats" in artifacts
        
        # Verify extensions were exercised
        ext_stats = artifacts["extension_stats"]
        assert any(stat["execution_count"] > 0 for stat in ext_stats.values())


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    # Run tests with asyncio support
    run_tests()
