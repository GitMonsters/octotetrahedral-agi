"""
Phase 9 (Continued): Advanced Model Routing & Fallback Strategies
==================================================================

Implements intelligent request routing with:
- Cost optimization
- Latency-aware routing
- Quality assurance routing
- Load balancing
- Intelligent fallback strategies
- Request prioritization
"""

import time
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ROUTING STRATEGIES
# ============================================================================

class RoutingStrategy(Enum):
    """Available routing strategies"""
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    QUALITY_FIRST = "quality_first"
    LOAD_BALANCED = "load_balanced"
    TIER_BASED = "tier_based"
    ROUND_ROBIN = "round_robin"
    ADAPTIVE = "adaptive"


class FallbackStrategy(Enum):
    """Fallback behavior when primary model fails"""
    IMMEDIATE = "immediate"              # Try next model immediately
    EXPONENTIAL_BACKOFF = "exponential"  # Wait with exponential backoff
    LINEAR_BACKOFF = "linear"            # Wait with linear backoff
    CIRCUIT_BREAKER = "circuit_breaker"  # Track failures, break circuit
    DISABLED = "disabled"                # No fallback


@dataclass
class RequestContext:
    """Context information for routing decisions"""
    request_id: str
    prompt: str
    priority: int = 5  # 1-10, higher is more important
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None
    min_quality: float = 0.7  # 0.0-1.0
    preferred_provider: Optional[str] = None
    allow_fallback: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    model_id: str
    reason: str
    estimated_cost: float
    estimated_latency: float
    confidence_score: float


@dataclass
class ModelMetrics:
    """Performance metrics for a model"""
    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: List[float] = field(default_factory=list)
    error_count_window: int = 0  # Errors in last window
    last_error: Optional[datetime] = None
    circuit_breaker_open: bool = False
    circuit_breaker_opened_at: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency(self) -> float:
        """Calculate average latency"""
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies)
    
    @property
    def p95_latency(self) -> float:
        """Calculate 95th percentile latency"""
        if len(self.latencies) < 2:
            return self.avg_latency
        return statistics.quantiles(self.latencies, n=20)[18]  # 95th percentile
    
    @property
    def avg_cost_per_request(self) -> float:
        """Calculate average cost per request"""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests
    
    def should_retry_circuit_breaker(self) -> bool:
        """Check if circuit breaker should attempt retry"""
        if not self.circuit_breaker_open:
            return False
        
        # Half-open state: retry after 60 seconds
        if self.circuit_breaker_opened_at:
            elapsed = (datetime.now() - self.circuit_breaker_opened_at).total_seconds()
            return elapsed >= 60.0
        
        return False
    
    def reset_metrics(self) -> None:
        """Reset metrics for new time window"""
        self.error_count_window = 0
        self.last_error = None


# ============================================================================
# ADVANCED ROUTER WITH MULTIPLE STRATEGIES
# ============================================================================

class AdvancedModelRouter:
    """Intelligent router with multiple strategies and fallback logic"""
    
    def __init__(self, model_configs: List[Dict[str, Any]]):
        self.model_configs = model_configs
        self.metrics: Dict[str, ModelMetrics] = {}
        self.routing_history: List[Tuple[RequestContext, RoutingDecision]] = []
        self.strategy = RoutingStrategy.ADAPTIVE
        self.fallback_strategy = FallbackStrategy.EXPONENTIAL_BACKOFF
        self.max_retries = 3
        self.error_threshold = 5  # Errors before circuit breaker opens
        self.circuit_breaker_window = 60  # seconds
        
        # Initialize metrics for each model
        for config in model_configs:
            self.metrics[config['model_id']] = ModelMetrics(model_id=config['model_id'])
    
    def _apply_cost_optimization(self, context: RequestContext) -> RoutingDecision:
        """Choose cheapest model that meets constraints"""
        logger.info(f"Applying COST_OPTIMIZED routing for {context.request_id}")
        
        candidates = []
        for config in self.model_configs:
            if not config.get('enabled', True):
                continue
            
            # Check constraints
            if context.max_cost and config.get('cost_per_request', 0) > context.max_cost:
                continue
            if context.max_latency and config.get('avg_latency', 0) > context.max_latency:
                continue
            
            candidates.append(config)
        
        if not candidates:
            candidates = self.model_configs
        
        # Sort by cost
        candidates.sort(key=lambda c: c.get('cost_per_request', 0))
        
        best = candidates[0]
        return RoutingDecision(
            model_id=best['model_id'],
            reason="Lowest cost model meeting constraints",
            estimated_cost=best.get('cost_per_request', 0),
            estimated_latency=best.get('avg_latency', 0),
            confidence_score=0.9
        )
    
    def _apply_latency_optimization(self, context: RequestContext) -> RoutingDecision:
        """Choose fastest model that meets constraints"""
        logger.info(f"Applying LATENCY_OPTIMIZED routing for {context.request_id}")
        
        candidates = []
        for config in self.model_configs:
            if not config.get('enabled', True):
                continue
            
            # Check constraints
            if context.max_cost and config.get('cost_per_request', 0) > context.max_cost:
                continue
            if context.max_latency and config.get('avg_latency', 0) > context.max_latency:
                continue
            
            candidates.append(config)
        
        if not candidates:
            candidates = self.model_configs
        
        # Sort by latency
        candidates.sort(key=lambda c: c.get('avg_latency', 0))
        
        best = candidates[0]
        return RoutingDecision(
            model_id=best['model_id'],
            reason="Fastest model meeting constraints",
            estimated_cost=best.get('cost_per_request', 0),
            estimated_latency=best.get('avg_latency', 0),
            confidence_score=0.9
        )
    
    def _apply_quality_first(self, context: RequestContext) -> RoutingDecision:
        """Choose model with highest quality/success rate"""
        logger.info(f"Applying QUALITY_FIRST routing for {context.request_id}")
        
        candidates = []
        for config in self.model_configs:
            if not config.get('enabled', True):
                continue
            
            metrics = self.metrics[config['model_id']]
            if metrics.success_rate < context.min_quality:
                continue
            
            candidates.append((config, metrics.success_rate))
        
        if not candidates:
            candidates = [(c, 0.5) for c in self.model_configs]
        
        # Sort by success rate
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best = candidates[0][0]
        metrics = self.metrics[best['model_id']]
        
        return RoutingDecision(
            model_id=best['model_id'],
            reason=f"Highest quality model (success rate: {metrics.success_rate:.2%})",
            estimated_cost=best.get('cost_per_request', 0),
            estimated_latency=best.get('avg_latency', 0),
            confidence_score=metrics.success_rate
        )
    
    def _apply_load_balanced(self, context: RequestContext) -> RoutingDecision:
        """Distribute load evenly across models"""
        logger.info(f"Applying LOAD_BALANCED routing for {context.request_id}")
        
        # Find model with fewest requests
        candidates = [
            (config, self.metrics[config['model_id']].total_requests)
            for config in self.model_configs
            if config.get('enabled', True)
        ]
        
        if not candidates:
            candidates = [(c, 0) for c in self.model_configs]
        
        candidates.sort(key=lambda x: x[1])
        
        best = candidates[0][0]
        return RoutingDecision(
            model_id=best['model_id'],
            reason="Lowest request count for load balancing",
            estimated_cost=best.get('cost_per_request', 0),
            estimated_latency=best.get('avg_latency', 0),
            confidence_score=0.8
        )
    
    def _apply_tier_based(self, context: RequestContext) -> RoutingDecision:
        """Route based on request priority and model tier"""
        logger.info(f"Applying TIER_BASED routing for {context.request_id}")
        
        # High priority -> ultra tier, Low priority -> efficient tier
        if context.priority >= 8:
            tier = "ultra"
        elif context.priority >= 5:
            tier = "advanced"
        else:
            tier = "efficient"
        
        candidates = [
            config for config in self.model_configs
            if config.get('tier') == tier and config.get('enabled', True)
        ]
        
        if not candidates:
            candidates = self.model_configs
        
        best = candidates[0]
        return RoutingDecision(
            model_id=best['model_id'],
            reason=f"Selected {tier} tier model for priority {context.priority}",
            estimated_cost=best.get('cost_per_request', 0),
            estimated_latency=best.get('avg_latency', 0),
            confidence_score=0.85
        )
    
    def _apply_round_robin(self, context: RequestContext) -> RoutingDecision:
        """Simple round-robin routing"""
        logger.info(f"Applying ROUND_ROBIN routing for {context.request_id}")
        
        enabled = [c for c in self.model_configs if c.get('enabled', True)]
        if not enabled:
            enabled = self.model_configs
        
        # Use history to determine next model
        idx = len(self.routing_history) % len(enabled)
        best = enabled[idx]
        
        return RoutingDecision(
            model_id=best['model_id'],
            reason="Round-robin selection",
            estimated_cost=best.get('cost_per_request', 0),
            estimated_latency=best.get('avg_latency', 0),
            confidence_score=0.7
        )
    
    def _apply_adaptive(self, context: RequestContext) -> RoutingDecision:
        """Adaptive routing based on context and metrics"""
        logger.info(f"Applying ADAPTIVE routing for {context.request_id}")
        
        # Use cost optimization if budget constrained
        if context.max_cost:
            return self._apply_cost_optimization(context)
        
        # Use latency optimization for time-sensitive requests
        if context.max_latency or context.priority >= 8:
            return self._apply_latency_optimization(context)
        
        # Use quality first for standard requests
        return self._apply_quality_first(context)
    
    async def route_request(
        self,
        context: RequestContext,
        strategy: Optional[RoutingStrategy] = None
    ) -> RoutingDecision:
        """Route request using specified strategy"""
        strategy = strategy or self.strategy
        
        # Route based on strategy
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            decision = self._apply_cost_optimization(context)
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            decision = self._apply_latency_optimization(context)
        elif strategy == RoutingStrategy.QUALITY_FIRST:
            decision = self._apply_quality_first(context)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            decision = self._apply_load_balanced(context)
        elif strategy == RoutingStrategy.TIER_BASED:
            decision = self._apply_tier_based(context)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            decision = self._apply_round_robin(context)
        elif strategy == RoutingStrategy.ADAPTIVE:
            decision = self._apply_adaptive(context)
        else:
            decision = self._apply_adaptive(context)
        
        # Log decision
        self.routing_history.append((context, decision))
        logger.info(f"Routed {context.request_id} to {decision.model_id}: {decision.reason}")
        
        return decision
    
    async def execute_with_fallback(
        self,
        context: RequestContext,
        execute_fn,
        strategy: Optional[RoutingStrategy] = None
    ) -> Optional[Any]:
        """Execute request with fallback logic"""
        retry_count = 0
        backoff_delay = 1.0
        
        while retry_count < self.max_retries:
            # Get routing decision
            decision = await self.route_request(context, strategy)
            model_id = decision.model_id
            metrics = self.metrics[model_id]
            
            # Update metrics
            metrics.total_requests += 1
            
            try:
                # Execute request
                start_time = time.time()
                result = await execute_fn(model_id, context)
                latency = (time.time() - start_time) * 1000
                
                # Record success
                metrics.successful_requests += 1
                metrics.latencies.append(latency)
                metrics.error_count_window = 0
                metrics.circuit_breaker_open = False
                
                logger.info(f"Request {context.request_id} succeeded on {model_id} ({latency:.2f}ms)")
                return result
                
            except Exception as e:
                logger.warning(f"Request {context.request_id} failed on {model_id}: {e}")
                
                metrics.failed_requests += 1
                metrics.error_count_window += 1
                metrics.last_error = datetime.now()
                
                # Circuit breaker logic
                if metrics.error_count_window >= self.error_threshold:
                    metrics.circuit_breaker_open = True
                    metrics.circuit_breaker_opened_at = datetime.now()
                    logger.error(f"Circuit breaker opened for {model_id}")
                
                # Fallback handling
                if not context.allow_fallback or retry_count >= self.max_retries - 1:
                    logger.error(f"Request {context.request_id} failed after {retry_count + 1} attempts")
                    return None
                
                # Apply fallback delay
                if self.fallback_strategy == FallbackStrategy.EXPONENTIAL_BACKOFF:
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2
                elif self.fallback_strategy == FallbackStrategy.LINEAR_BACKOFF:
                    await asyncio.sleep(backoff_delay)
                    backoff_delay += 1.0
                
                retry_count += 1
        
        return None
    
    def get_router_statistics(self) -> Dict[str, Any]:
        """Get comprehensive router statistics"""
        total_requests = sum(m.total_requests for m in self.metrics.values())
        total_cost = sum(m.total_cost for m in self.metrics.values())
        total_success = sum(m.successful_requests for m in self.metrics.values())
        
        return {
            'total_requests': total_requests,
            'total_successful': total_success,
            'success_rate': total_success / total_requests if total_requests > 0 else 1.0,
            'total_cost': total_cost,
            'routing_history_size': len(self.routing_history),
            'models': {
                model_id: {
                    'total_requests': metrics.total_requests,
                    'success_rate': metrics.success_rate,
                    'avg_latency': metrics.avg_latency,
                    'p95_latency': metrics.p95_latency,
                    'avg_cost_per_request': metrics.avg_cost_per_request,
                    'circuit_breaker_open': metrics.circuit_breaker_open,
                }
                for model_id, metrics in self.metrics.items()
            }
        }


# ============================================================================
# DEMO AND TESTING
# ============================================================================

async def demo_advanced_routing():
    """Demonstrate advanced routing strategies"""
    print("=" * 70)
    print("Advanced Model Routing & Fallback Strategies Demo")
    print("=" * 70)
    
    # Define model configurations with metrics
    model_configs = [
        {
            'model_id': 'gpt-4',
            'provider': 'openai',
            'tier': 'ultra',
            'cost_per_request': 0.04,
            'avg_latency': 2500,
            'enabled': True,
        },
        {
            'model_id': 'gpt-3.5-turbo',
            'provider': 'openai',
            'tier': 'advanced',
            'cost_per_request': 0.002,
            'avg_latency': 1500,
            'enabled': True,
        },
        {
            'model_id': 'claude-3-opus',
            'provider': 'anthropic',
            'tier': 'ultra',
            'cost_per_request': 0.090,
            'avg_latency': 2000,
            'enabled': True,
        },
        {
            'model_id': 'claude-3-sonnet',
            'provider': 'anthropic',
            'tier': 'advanced',
            'cost_per_request': 0.018,
            'avg_latency': 1200,
            'enabled': True,
        },
        {
            'model_id': 'gemini-pro',
            'provider': 'google',
            'tier': 'advanced',
            'cost_per_request': 0.0,
            'avg_latency': 1800,
            'enabled': True,
        },
    ]
    
    # Create router
    router = AdvancedModelRouter(model_configs)
    
    print("\n1. Available Models:")
    print("-" * 70)
    for config in model_configs:
        print(f"   • {config['model_id']} ({config['provider']}) - "
              f"${config['cost_per_request']:.4f}, {config['avg_latency']}ms")
    
    # Test different routing strategies
    test_contexts = [
        RequestContext(
            request_id="req-001",
            prompt="Explain quantum computing",
            priority=9,
            max_latency=2000,
            preferred_provider="openai"
        ),
        RequestContext(
            request_id="req-002",
            prompt="Summarize a document",
            priority=5,
            max_cost=0.01,
        ),
        RequestContext(
            request_id="req-003",
            prompt="Generate creative content",
            priority=3,
        ),
    ]
    
    strategies = [
        RoutingStrategy.LATENCY_OPTIMIZED,
        RoutingStrategy.COST_OPTIMIZED,
        RoutingStrategy.ADAPTIVE,
    ]
    
    print("\n2. Testing Routing Strategies:")
    print("-" * 70)
    
    for i, context in enumerate(test_contexts):
        print(f"\n   Request {i+1}: {context.prompt[:40]}...")
        print(f"   Priority: {context.priority}, Max Latency: {context.max_latency}ms, Max Cost: ${context.max_cost}")
        
        for strategy in strategies:
            router.strategy = strategy
            decision = await router.route_request(context, strategy)
            print(f"   • {strategy.value}: {decision.model_id} "
                  f"(confidence: {decision.confidence_score:.2%}, cost: ${decision.estimated_cost:.4f})")
    
    # Simulate execution with metrics updates
    print("\n3. Simulating Execution with Metrics:")
    print("-" * 70)
    
    async def mock_execute(model_id: str, context: RequestContext) -> str:
        """Mock execution function"""
        await asyncio.sleep(0.01)  # Simulate execution
        if model_id == "gpt-4":
            return f"Response from {model_id}"
        return f"Response from {model_id}"
    
    for context in test_contexts[:2]:
        router.strategy = RoutingStrategy.ADAPTIVE
        result = await router.execute_with_fallback(context, mock_execute)
        print(f"\n   Request {context.request_id}: {'SUCCESS' if result else 'FAILED'}")
    
    # Get statistics
    print("\n4. Router Statistics:")
    print("-" * 70)
    stats = router.get_router_statistics()
    
    print(f"\n   Total Requests: {stats['total_requests']}")
    print(f"   Total Successful: {stats['total_successful']}")
    print(f"   Overall Success Rate: {stats['success_rate']:.2%}")
    print(f"   Total Cost: ${stats['total_cost']:.4f}")
    
    print("\n   Per-Model Statistics:")
    for model_id, model_stats in stats['models'].items():
        print(f"\n   {model_id}:")
        print(f"     • Requests: {model_stats['total_requests']}")
        print(f"     • Success Rate: {model_stats['success_rate']:.2%}")
        print(f"     • Avg Latency: {model_stats['avg_latency']:.0f}ms")
        print(f"     • P95 Latency: {model_stats['p95_latency']:.0f}ms")
        print(f"     • Cost/Request: ${model_stats['avg_cost_per_request']:.4f}")
        print(f"     • Circuit Breaker: {'OPEN' if model_stats['circuit_breaker_open'] else 'CLOSED'}")
    
    print("\n" + "=" * 70)
    print("Advanced Routing Demo Complete ✓")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_advanced_routing())
