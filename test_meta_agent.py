"""
Test suite for NGVTMetaAgent
Tests configuration optimization, parameter synthesis, and evaluation
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from ngvt_meta_agent import (
    NGVTMetaAgent,
    ConfigurationSpace,
    ConfigurationEvaluator,
    SimpleConfigEvaluator,
    ConfigurationSynthesizer,
    ParameterSpec,
    ParameterType,
    OptimizationStrategy,
    ConfigCandidate,
)


class TestParameterSpec:
    """Tests for ParameterSpec dataclass"""
    
    def test_int_parameter_creation(self):
        """Test creating integer parameter"""
        param = ParameterSpec(
            name="max_iterations",
            param_type=ParameterType.INT,
            default_value=10,
            min_value=1,
            max_value=100,
        )
        assert param.name == "max_iterations"
        assert param.param_type == ParameterType.INT
        assert param.default_value == 10
        assert param.min_value == 1
        assert param.max_value == 100
    
    def test_float_parameter_creation(self):
        """Test creating float parameter"""
        param = ParameterSpec(
            name="temperature",
            param_type=ParameterType.FLOAT,
            default_value=0.7,
            min_value=0.0,
            max_value=2.0,
        )
        assert param.param_type == ParameterType.FLOAT
        assert param.default_value == 0.7
    
    def test_choice_parameter_creation(self):
        """Test creating choice parameter"""
        param = ParameterSpec(
            name="model",
            param_type=ParameterType.CHOICE,
            default_value="claude",
            possible_values=["claude", "gpt", "palm"],
        )
        assert param.param_type == ParameterType.CHOICE
        assert param.possible_values == ["claude", "gpt", "palm"]


class TestConfigurationSpace:
    """Tests for ConfigurationSpace"""
    
    def test_create_config_space(self):
        """Test creating configuration space"""
        params = [
            ParameterSpec(
                name="max_iterations",
                param_type=ParameterType.INT,
                default_value=10,
                min_value=1,
                max_value=50,
            ),
            ParameterSpec(
                name="temperature",
                param_type=ParameterType.FLOAT,
                default_value=0.7,
                min_value=0.0,
                max_value=2.0,
            ),
        ]
        
        space = ConfigurationSpace(params)
        assert len(space.parameters) == 2
        assert "max_iterations" in space.parameters
    
    def test_get_default_config(self):
        """Test getting default configuration"""
        params = [
            ParameterSpec(
                name="param1",
                param_type=ParameterType.INT,
                default_value=10,
                min_value=1,
                max_value=50,
            ),
            ParameterSpec(
                name="param2",
                param_type=ParameterType.FLOAT,
                default_value=0.7,
                min_value=0.0,
                max_value=1.0,
            ),
        ]
        
        space = ConfigurationSpace(params)
        default = space.get_default_config()
        
        assert default["param1"] == 10
        assert default["param2"] == 0.7
    
    def test_generate_random_config(self):
        """Test generating random configuration"""
        params = [
            ParameterSpec(
                name="max_iterations",
                param_type=ParameterType.INT,
                default_value=10,
                min_value=5,
                max_value=20,
            ),
        ]
        
        space = ConfigurationSpace(params)
        random_config = space.generate_random_config()
        
        assert "max_iterations" in random_config
        assert 5 <= random_config["max_iterations"] <= 20
    
    def test_generate_grid_configs_int(self):
        """Test generating grid search configs for int parameter"""
        params = [
            ParameterSpec(
                name="batch_size",
                param_type=ParameterType.INT,
                default_value=16,
                min_value=8,
                max_value=64,
            ),
        ]
        
        space = ConfigurationSpace(params)
        configs = space.generate_grid_configs("batch_size")
        
        assert len(configs) > 0
        values = [c["batch_size"] for c in configs]
        assert all(8 <= v <= 64 for v in values)
    
    def test_generate_grid_configs_choice(self):
        """Test generating grid search configs for choice parameter"""
        params = [
            ParameterSpec(
                name="model",
                param_type=ParameterType.CHOICE,
                default_value="claude",
                possible_values=["claude", "gpt", "palm"],
            ),
        ]
        
        space = ConfigurationSpace(params)
        configs = space.generate_grid_configs("model")
        
        assert len(configs) == 3
        values = [c["model"] for c in configs]
        assert set(values) == {"claude", "gpt", "palm"}


class TestConfigCandidate:
    """Tests for ConfigCandidate"""
    
    def test_create_candidate(self):
        """Test creating candidate configuration"""
        candidate = ConfigCandidate(
            id="config_001",
            config={"param1": 10, "param2": 0.7},
            tested=True,
            performance_score=0.85,
            execution_latency_ms=150.5,
        )
        
        assert candidate.id == "config_001"
        assert candidate.tested is True
        assert candidate.performance_score == 0.85
        assert candidate.execution_latency_ms == 150.5


class TestSimpleConfigEvaluator:
    """Tests for SimpleConfigEvaluator"""
    
    @pytest.mark.asyncio
    async def test_evaluate_config(self):
        """Test evaluating a configuration"""
        evaluator = SimpleConfigEvaluator()
        
        config = {
            "max_iterations": 10,
            "temperature": 0.7,
            "batch_size": 16,
        }
        
        score = await evaluator.evaluate(config)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting detailed metrics"""
        evaluator = SimpleConfigEvaluator()
        
        config = {
            "max_iterations": 10,
            "temperature": 0.7,
        }
        
        metrics = await evaluator.get_metrics(config)
        
        assert "performance_score" in metrics
        assert "execution_latency_ms" in metrics
        assert "memory_usage_mb" in metrics
        assert "success_rate" in metrics
    
    @pytest.mark.asyncio
    async def test_evaluate_multiple_configs(self):
        """Test evaluating multiple configurations"""
        evaluator = SimpleConfigEvaluator()
        
        configs = [
            {"temperature": 0.5},
            {"temperature": 0.7},
            {"temperature": 1.0},
        ]
        
        scores = []
        for config in configs:
            score = await evaluator.evaluate(config)
            scores.append(score)
        
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestConfigurationSynthesizer:
    """Tests for ConfigurationSynthesizer"""
    
    def test_create_orchestrator_parameters(self):
        """Test creating orchestrator parameter specs"""
        params = ConfigurationSynthesizer.create_orchestrator_parameters()
        
        assert len(params) > 0
        names = {p.name for p in params}
        assert "max_iterations" in names
        assert "temperature" in names
    
    def test_create_inference_parameters(self):
        """Test creating inference parameter specs"""
        params = ConfigurationSynthesizer.create_inference_parameters()
        
        assert len(params) > 0
        names = {p.name for p in params}
        assert "inference_model" in names
        assert "max_tokens" in names
    
    def test_create_memory_parameters(self):
        """Test creating memory parameter specs"""
        params = ConfigurationSynthesizer.create_memory_parameters()
        
        assert len(params) > 0
        names = {p.name for p in params}
        assert "session_weight" in names
        assert "compression_threshold" in names


class TestNGVTMetaAgent:
    """Tests for NGVTMetaAgent"""
    
    @pytest.mark.asyncio
    async def test_meta_agent_creation(self):
        """Test creating meta-agent"""
        params = ConfigurationSynthesizer.create_orchestrator_parameters()
        config_space = ConfigurationSpace(params)
        evaluator = SimpleConfigEvaluator()
        
        meta_agent = NGVTMetaAgent(
            config_space=config_space,
            evaluator=evaluator,
            strategy=OptimizationStrategy.RANDOM_SEARCH,
        )
        
        assert meta_agent.config_space is not None
        assert meta_agent.evaluator is not None
        assert meta_agent.strategy == OptimizationStrategy.RANDOM_SEARCH
    
    @pytest.mark.asyncio
    async def test_generate_candidate_random(self):
        """Test generating random candidate"""
        params = [
            ParameterSpec(
                name="param1",
                param_type=ParameterType.INT,
                default_value=10,
                min_value=1,
                max_value=50,
            ),
        ]
        
        config_space = ConfigurationSpace(params)
        evaluator = SimpleConfigEvaluator()
        meta_agent = NGVTMetaAgent(
            config_space=config_space,
            evaluator=evaluator,
            strategy=OptimizationStrategy.RANDOM_SEARCH,
        )
        
        candidate = meta_agent._generate_candidate(1)
        
        assert "param1" in candidate
        assert 1 <= candidate["param1"] <= 50
    
    @pytest.mark.asyncio
    async def test_optimize_convergence(self):
        """Test optimization convergence"""
        params = ConfigurationSynthesizer.create_orchestrator_parameters()[:2]
        config_space = ConfigurationSpace(params)
        evaluator = SimpleConfigEvaluator()
        
        meta_agent = NGVTMetaAgent(
            config_space=config_space,
            evaluator=evaluator,
            strategy=OptimizationStrategy.RANDOM_SEARCH,
        )
        
        result = await meta_agent.optimize(
            max_iterations=10,
            timeout_seconds=30,
            verbose=False,
        )
        
        assert result.best_config is not None
        assert result.best_score > 0
        assert len(result.convergence_curve) > 0
        assert result.iterations > 0
    
    @pytest.mark.asyncio
    async def test_optimize_bayesian_strategy(self):
        """Test optimization with Bayesian strategy"""
        params = ConfigurationSynthesizer.create_orchestrator_parameters()[:1]
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
        
        assert result.best_score > 0
        assert len(result.config_history) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_evolutionary_strategy(self):
        """Test optimization with evolutionary strategy"""
        params = ConfigurationSynthesizer.create_orchestrator_parameters()[:2]
        config_space = ConfigurationSpace(params)
        evaluator = SimpleConfigEvaluator()
        
        meta_agent = NGVTMetaAgent(
            config_space=config_space,
            evaluator=evaluator,
            strategy=OptimizationStrategy.EVOLUTIONARY,
        )
        
        result = await meta_agent.optimize(
            max_iterations=8,
            timeout_seconds=30,
            verbose=False,
        )
        
        assert result.best_score >= 0
        assert result.candidates_evaluated > 0
    
    @pytest.mark.asyncio
    async def test_get_best_config(self):
        """Test getting best configuration"""
        params = ConfigurationSynthesizer.create_orchestrator_parameters()[:2]
        config_space = ConfigurationSpace(params)
        evaluator = SimpleConfigEvaluator()
        
        meta_agent = NGVTMetaAgent(
            config_space=config_space,
            evaluator=evaluator,
        )
        
        await meta_agent.optimize(max_iterations=5, verbose=False)
        best = meta_agent.get_best_config()
        
        assert best is not None
        assert isinstance(best, dict)
    
    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test getting optimization statistics"""
        params = ConfigurationSynthesizer.create_orchestrator_parameters()[:2]
        config_space = ConfigurationSpace(params)
        evaluator = SimpleConfigEvaluator()
        
        meta_agent = NGVTMetaAgent(
            config_space=config_space,
            evaluator=evaluator,
        )
        
        await meta_agent.optimize(max_iterations=5, verbose=False)
        stats = meta_agent.get_statistics()
        
        assert "best_score" in stats
        assert "candidates_evaluated" in stats
        assert "convergence_path" in stats
        assert stats["candidates_evaluated"] > 0


class TestOptimizationStrategies:
    """Tests for different optimization strategies"""
    
    @pytest.mark.asyncio
    async def test_grid_search_strategy(self):
        """Test grid search optimization"""
        params = ConfigurationSynthesizer.create_orchestrator_parameters()[:1]
        config_space = ConfigurationSpace(params)
        evaluator = SimpleConfigEvaluator()
        
        meta_agent = NGVTMetaAgent(
            config_space=config_space,
            evaluator=evaluator,
            strategy=OptimizationStrategy.GRID_SEARCH,
        )
        
        result = await meta_agent.optimize(
            max_iterations=5,
            timeout_seconds=30,
            verbose=False,
        )
        
        assert result.best_score >= 0
    
    @pytest.mark.asyncio
    async def test_all_strategies_convergence(self):
        """Test that all strategies can converge"""
        strategies = [
            OptimizationStrategy.RANDOM_SEARCH,
            OptimizationStrategy.BAYESIAN,
            OptimizationStrategy.EVOLUTIONARY,
        ]
        
        for strategy in strategies:
            params = ConfigurationSynthesizer.create_orchestrator_parameters()[:1]
            config_space = ConfigurationSpace(params)
            evaluator = SimpleConfigEvaluator()
            
            meta_agent = NGVTMetaAgent(
                config_space=config_space,
                evaluator=evaluator,
                strategy=strategy,
            )
            
            result = await meta_agent.optimize(
                max_iterations=5,
                timeout_seconds=30,
                verbose=False,
            )
            
            assert result.best_score >= 0, f"Strategy {strategy.value} failed to converge"


class TestMetaAgentIntegration:
    """Integration tests for meta-agent"""
    
    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self):
        """Test complete optimization workflow"""
        # Setup
        params = ConfigurationSynthesizer.create_orchestrator_parameters()
        config_space = ConfigurationSpace(params)
        evaluator = SimpleConfigEvaluator()
        
        meta_agent = NGVTMetaAgent(
            config_space=config_space,
            evaluator=evaluator,
            strategy=OptimizationStrategy.BAYESIAN,
        )
        
        # Optimize
        result = await meta_agent.optimize(
            max_iterations=10,
            timeout_seconds=30,
            verbose=False,
        )
        
        # Verify
        assert result is not None
        assert result.best_config is not None
        assert result.best_score > 0
        assert result.iterations > 0
        assert len(result.convergence_curve) > 0
        assert len(result.config_history) > 0
        
        # Check convergence
        curve = result.convergence_curve
        final_improvement = curve[-1] - curve[0]
        assert final_improvement >= -0.1  # Allow small degradation due to randomness


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
