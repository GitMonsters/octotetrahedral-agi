"""
NGVT Meta-Agent Configuration Synthesis
Automated configuration optimization and tuning system
Discovers optimal hyperparameters for orchestration and inference
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import asyncio
from abc import ABC, abstractmethod


class ParameterType(Enum):
    """Types of parameters that can be tuned"""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CHOICE = "choice"


class OptimizationStrategy(Enum):
    """Parameter optimization strategies"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"


@dataclass
class ParameterSpec:
    """Specification for a parameter to optimize"""
    name: str
    param_type: ParameterType
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    possible_values: Optional[List[Any]] = None
    importance: float = 1.0  # Relative importance for optimization


@dataclass
class ConfigCandidate:
    """A candidate configuration to evaluate"""
    id: str
    config: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tested: bool = False
    performance_score: float = 0.0
    execution_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    success_rate: float = 0.0


@dataclass
class OptimizationResult:
    """Result of optimization run"""
    best_config: Dict[str, Any]
    best_score: float
    candidates_evaluated: int
    iterations: int
    total_time_seconds: float
    config_history: List[ConfigCandidate]
    convergence_curve: List[float]


class ConfigurationSpace:
    """Manages the parameter optimization space"""
    
    def __init__(self, parameters: List[ParameterSpec]):
        self.parameters = {p.name: p for p in parameters}
        self.parameter_list = parameters
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {p.name: p.default_value for p in self.parameter_list}
    
    def generate_random_config(self) -> Dict[str, Any]:
        """Generate random configuration within parameter space"""
        config = {}
        
        for param in self.parameter_list:
            if param.param_type == ParameterType.INT:
                value = self._random_int(param)
            elif param.param_type == ParameterType.FLOAT:
                value = self._random_float(param)
            elif param.param_type == ParameterType.BOOL:
                value = self._random_bool()
            elif param.param_type == ParameterType.CHOICE:
                value = self._random_choice(param)
            else:
                value = param.default_value
            
            config[param.name] = value
        
        return config
    
    def generate_grid_configs(self, param_name: str) -> List[Dict[str, Any]]:
        """Generate grid search configs for single parameter"""
        param = self.parameters[param_name]
        base_config = self.get_default_config()
        configs = []
        
        if param.param_type == ParameterType.CHOICE and param.possible_values:
            for value in param.possible_values:
                config = base_config.copy()
                config[param_name] = value
                configs.append(config)
        elif param.param_type == ParameterType.INT:
            step = max(1, (param.max_value - param.min_value) // 5)
            for value in range(param.min_value, param.max_value + 1, step):
                config = base_config.copy()
                config[param_name] = value
                configs.append(config)
        elif param.param_type == ParameterType.FLOAT:
            step = (param.max_value - param.min_value) / 5
            current = param.min_value
            while current <= param.max_value:
                config = base_config.copy()
                config[param_name] = current
                configs.append(config)
                current += step
        
        return configs
    
    @staticmethod
    def _random_int(param: ParameterSpec) -> int:
        import random
        return random.randint(param.min_value, param.max_value)
    
    @staticmethod
    def _random_float(param: ParameterSpec) -> float:
        import random
        return random.uniform(param.min_value, param.max_value)
    
    @staticmethod
    def _random_bool() -> bool:
        import random
        return random.choice([True, False])
    
    @staticmethod
    def _random_choice(param: ParameterSpec) -> Any:
        import random
        return random.choice(param.possible_values)


class ConfigurationEvaluator(ABC):
    """Base class for evaluating configurations"""
    
    @abstractmethod
    async def evaluate(self, config: Dict[str, Any]) -> float:
        """Evaluate configuration and return performance score (0-1)"""
        pass
    
    @abstractmethod
    async def get_metrics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed metrics for configuration"""
        pass


class SimpleConfigEvaluator(ConfigurationEvaluator):
    """Simple evaluator for testing - simulates configuration evaluation"""
    
    async def evaluate(self, config: Dict[str, Any]) -> float:
        """Simulate performance evaluation"""
        # Score based on config parameters
        score = 0.5
        
        if "max_iterations" in config:
            # More iterations = slightly better (but diminishing returns)
            score += min(0.2, config["max_iterations"] / 50)
        
        if "temperature" in config:
            # Temperature around 0.7 is optimal
            temp_diff = abs(config["temperature"] - 0.7)
            score += max(0, 0.2 - temp_diff * 0.2)
        
        if "memory_compression_enabled" in config and config["memory_compression_enabled"]:
            score += 0.1
        
        if "batch_size" in config:
            # Larger batch sizes up to 32 are better
            score += min(0.1, config["batch_size"] / 320)
        
        # Add some randomness to simulate real evaluation
        import random
        score += random.uniform(-0.05, 0.05)
        
        return min(1.0, max(0.0, score))
    
    async def get_metrics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return detailed metrics"""
        import random
        score = await self.evaluate(config)
        
        return {
            "performance_score": score,
            "execution_latency_ms": random.uniform(10, 1000),
            "memory_usage_mb": random.uniform(50, 500),
            "success_rate": 0.8 + random.uniform(0, 0.2),
            "throughput_ops_sec": random.uniform(100, 1000),
        }


class NGVTMetaAgent:
    """
    Meta-agent for automated configuration optimization
    Discovers optimal hyperparameters for orchestration and inference
    """
    
    def __init__(
        self,
        config_space: ConfigurationSpace,
        evaluator: ConfigurationEvaluator,
        strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH,
    ):
        self.config_space = config_space
        self.evaluator = evaluator
        self.strategy = strategy
        self.candidates: List[ConfigCandidate] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: float = 0.0
        self.convergence_curve: List[float] = []
    
    async def optimize(
        self,
        max_iterations: int = 20,
        timeout_seconds: int = 300,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Main optimization loop - discover optimal configuration
        """
        start_time = time.time()
        iteration = 0
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting configuration optimization (strategy: {self.strategy.value})")
            print(f"{'='*70}\n")
        
        try:
            # Start with default configuration
            default_config = self.config_space.get_default_config()
            default_score = await self._evaluate_config(default_config)
            self.best_config = default_config
            self.best_score = default_score
            self.convergence_curve.append(default_score)
            
            if verbose:
                print(f"[*] Default config score: {default_score:.4f}\n")
            
            # Main optimization loop
            while iteration < max_iterations:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    if verbose:
                        print(f"[!] Timeout reached ({elapsed:.1f}s)")
                    break
                
                iteration += 1
                
                # Generate candidate configuration
                candidate_config = self._generate_candidate(iteration)
                
                # Evaluate candidate
                score = await self._evaluate_config(candidate_config)
                
                # Track convergence
                self.convergence_curve.append(score)
                
                # Update best if improved
                improved = False
                if score > self.best_score:
                    improvement = score - self.best_score
                    self.best_score = score
                    self.best_config = candidate_config.copy()
                    improved = True
                
                status = "↑ IMPROVED" if improved else "  -"
                if verbose:
                    print(f"[{iteration:2d}] Score: {score:.4f} | Best: {self.best_score:.4f} {status}")
            
            elapsed = time.time() - start_time
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"Optimization complete in {elapsed:.2f}s ({iteration} iterations)")
                print(f"Best score: {self.best_score:.4f}")
                print(f"Improvement: {(self.best_score - default_score):.4f}")
                print(f"{'='*70}\n")
            
            return OptimizationResult(
                best_config=self.best_config,
                best_score=self.best_score,
                candidates_evaluated=len(self.candidates),
                iterations=iteration,
                total_time_seconds=elapsed,
                config_history=self.candidates,
                convergence_curve=self.convergence_curve,
            )
        
        except Exception as e:
            if verbose:
                print(f"[ERROR] {str(e)}")
            raise
    
    def _generate_candidate(self, iteration: int) -> Dict[str, Any]:
        """Generate next candidate configuration"""
        if self.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return self.config_space.generate_random_config()
        
        elif self.strategy == OptimizationStrategy.GRID_SEARCH:
            # Grid search around best configuration
            if iteration <= len(self.config_space.parameter_list):
                param_name = self.config_space.parameter_list[iteration - 1].name
                configs = self.config_space.generate_grid_configs(param_name)
                idx = (iteration - 1) % len(configs)
                return configs[idx]
            return self.config_space.generate_random_config()
        
        elif self.strategy == OptimizationStrategy.BAYESIAN:
            # Simplified Bayesian approach - favor exploration near best
            config = self.config_space.generate_random_config()
            
            # With some probability, mutate from best config
            import random
            if random.random() < 0.6 and self.best_config:
                for param in self.config_space.parameter_list:
                    if random.random() < 0.3:
                        config[param.name] = self.best_config.get(
                            param.name, param.default_value
                        )
            
            return config
        
        elif self.strategy == OptimizationStrategy.EVOLUTIONARY:
            # Evolutionary approach - mutation around best
            if not self.best_config:
                return self.config_space.generate_random_config()
            
            config = self.best_config.copy()
            
            # Mutate 1-3 random parameters
            import random
            num_mutations = random.randint(1, 3)
            params_to_mutate = random.sample(
                self.config_space.parameter_list,
                min(num_mutations, len(self.config_space.parameter_list))
            )
            
            for param in params_to_mutate:
                if param.param_type == ParameterType.INT:
                    # Mutate by small amount
                    step = max(1, (param.max_value - param.min_value) // 10)
                    delta = random.randint(-step, step)
                    new_value = max(param.min_value, min(param.max_value, config[param.name] + delta))
                    config[param.name] = new_value
                
                elif param.param_type == ParameterType.FLOAT:
                    # Mutate by small amount
                    step = (param.max_value - param.min_value) / 10
                    delta = random.uniform(-step, step)
                    new_value = max(param.min_value, min(param.max_value, config[param.name] + delta))
                    config[param.name] = new_value
                
                elif param.param_type == ParameterType.CHOICE:
                    # Random choice
                    config[param.name] = self.config_space._random_choice(param)
            
            return config
        
        return self.config_space.generate_random_config()
    
    async def _evaluate_config(self, config: Dict[str, Any]) -> float:
        """Evaluate a configuration and track it"""
        candidate_id = f"config_{len(self.candidates):04d}"
        
        try:
            score = await self.evaluator.evaluate(config)
            metrics = await self.evaluator.get_metrics(config)
            
            candidate = ConfigCandidate(
                id=candidate_id,
                config=config.copy(),
                tested=True,
                performance_score=score,
                execution_latency_ms=metrics.get("execution_latency_ms", 0),
                memory_usage_mb=metrics.get("memory_usage_mb", 0),
                success_rate=metrics.get("success_rate", 0),
            )
            
            self.candidates.append(candidate)
            return score
        
        except Exception as e:
            # On error, return low score
            candidate = ConfigCandidate(
                id=candidate_id,
                config=config.copy(),
                tested=True,
                performance_score=0.0,
            )
            self.candidates.append(candidate)
            return 0.0
    
    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get the best configuration discovered"""
        return self.best_config
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        scores = [c.performance_score for c in self.candidates]
        
        return {
            "best_score": self.best_score,
            "worst_score": min(scores) if scores else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "candidates_evaluated": len(self.candidates),
            "convergence_path": self.convergence_curve,
            "best_config": self.best_config,
        }


class ConfigurationSynthesizer:
    """
    Synthesizes optimal configurations for different scenarios
    """
    
    @staticmethod
    def create_orchestrator_parameters() -> List[ParameterSpec]:
        """Create parameter specs for orchestrator tuning"""
        return [
            ParameterSpec(
                name="max_iterations",
                param_type=ParameterType.INT,
                default_value=10,
                min_value=3,
                max_value=50,
                importance=1.0,
            ),
            ParameterSpec(
                name="temperature",
                param_type=ParameterType.FLOAT,
                default_value=0.7,
                min_value=0.0,
                max_value=2.0,
                importance=1.0,
            ),
            ParameterSpec(
                name="max_context_tokens",
                param_type=ParameterType.INT,
                default_value=8000,
                min_value=1000,
                max_value=32000,
                importance=0.8,
            ),
            ParameterSpec(
                name="memory_compression_enabled",
                param_type=ParameterType.BOOL,
                default_value=True,
                importance=0.9,
            ),
            ParameterSpec(
                name="batch_size",
                param_type=ParameterType.INT,
                default_value=16,
                min_value=1,
                max_value=64,
                importance=0.7,
            ),
        ]
    
    @staticmethod
    def create_inference_parameters() -> List[ParameterSpec]:
        """Create parameter specs for inference tuning"""
        return [
            ParameterSpec(
                name="inference_model",
                param_type=ParameterType.CHOICE,
                default_value="claude",
                possible_values=["claude", "gpt", "palm", "llama"],
                importance=1.0,
            ),
            ParameterSpec(
                name="max_tokens",
                param_type=ParameterType.INT,
                default_value=500,
                min_value=50,
                max_value=4000,
                importance=0.9,
            ),
            ParameterSpec(
                name="top_p",
                param_type=ParameterType.FLOAT,
                default_value=0.9,
                min_value=0.0,
                max_value=1.0,
                importance=0.6,
            ),
            ParameterSpec(
                name="top_k",
                param_type=ParameterType.INT,
                default_value=40,
                min_value=1,
                max_value=100,
                importance=0.6,
            ),
        ]
    
    @staticmethod
    def create_memory_parameters() -> List[ParameterSpec]:
        """Create parameter specs for memory tuning"""
        return [
            ParameterSpec(
                name="session_weight",
                param_type=ParameterType.FLOAT,
                default_value=0.4,
                min_value=0.1,
                max_value=0.7,
                importance=0.8,
            ),
            ParameterSpec(
                name="entry_weight",
                param_type=ParameterType.FLOAT,
                default_value=0.35,
                min_value=0.1,
                max_value=0.7,
                importance=0.8,
            ),
            ParameterSpec(
                name="runnable_weight",
                param_type=ParameterType.FLOAT,
                default_value=0.25,
                min_value=0.1,
                max_value=0.7,
                importance=0.8,
            ),
            ParameterSpec(
                name="compression_threshold",
                param_type=ParameterType.FLOAT,
                default_value=0.8,
                min_value=0.5,
                max_value=0.95,
                importance=0.7,
            ),
        ]


async def demo_meta_agent():
    """Demo meta-agent optimization"""
    
    # Create parameter space
    parameters = ConfigurationSynthesizer.create_orchestrator_parameters()
    config_space = ConfigurationSpace(parameters)
    
    # Create simple evaluator
    evaluator = SimpleConfigEvaluator()
    
    # Create meta-agent
    meta_agent = NGVTMetaAgent(
        config_space=config_space,
        evaluator=evaluator,
        strategy=OptimizationStrategy.BAYESIAN,
    )
    
    # Run optimization
    result = await meta_agent.optimize(
        max_iterations=15,
        timeout_seconds=30,
        verbose=True,
    )
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\nBest Configuration:")
    for key, value in result.best_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nOptimization Stats:")
    print(f"  Best Score: {result.best_score:.4f}")
    print(f"  Candidates Evaluated: {result.candidates_evaluated}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Total Time: {result.total_time_seconds:.2f}s")
    
    print(f"\nTop 5 Configurations:")
    sorted_candidates = sorted(
        result.config_history,
        key=lambda c: c.performance_score,
        reverse=True
    )
    for i, candidate in enumerate(sorted_candidates[:5], 1):
        print(f"  {i}. Score: {candidate.performance_score:.4f}")
        print(f"     Config: {candidate.config}")
    
    print(f"\nConvergence Curve (sample):")
    step = max(1, len(result.convergence_curve) // 10)
    for i, score in enumerate(result.convergence_curve[::step]):
        print(f"  Iter {i*step}: {score:.4f}")
    
    return result


if __name__ == "__main__":
    asyncio.run(demo_meta_agent())
