"""
NGVT Compound Learning-Enhanced Orchestrator
Integrates compound learning system with unified orchestrator for intelligent
multi-model coordination and adaptive workflow optimization
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from ngvt_orchestrator import (
    NGVTUnifiedOrchestrator,
    OrchestratorConfig,
    Task,
    Action,
    ActionType,
    Extension,
    Observation,
)

from ngvt_compound_learning import (
    CompoundLearningEngine,
    CompoundIntegrationEngine,
    LearningExperience,
    IntegrationWorkflow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompoundLearningExtension(Extension):
    """
    Extension that enables compound learning within orchestration
    Captures and learns from orchestration actions and results
    """
    
    def __init__(self, config: OrchestratorConfig, 
                 learning_engine: CompoundLearningEngine):
        super().__init__("CompoundLearningExtension", config)
        self.learning_engine = learning_engine
    
    async def can_handle(self, action: Action) -> bool:
        """Can handle any action for learning purposes"""
        return True
    
    async def execute(self, action: Action) -> Observation:
        """Record action for learning analysis"""
        start_time = time.time()
        
        # Record the action context
        self.learning_engine.record_experience(
            LearningExperience(
                query=str(action.params),
                response=str(action.reasoning),
                latency_ms=0.0,
                success=True,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'action_type': action.type.value,
                    'reasoning': action.reasoning,
                }
            )
        )
        
        latency_ms = (time.time() - start_time) * 1000
        self.execution_count += 1
        self.total_latency_ms += latency_ms
        
        return Observation(
            action_type=action.type,
            success=True,
            result={'learning_recorded': True},
            latency_ms=latency_ms,
        )


class AdaptiveWorkflowOrchestrator(NGVTUnifiedOrchestrator):
    """
    Enhanced orchestrator that learns and adapts workflows using compound learning
    Automatically optimizes multi-model execution patterns
    """
    
    def __init__(self, config: OrchestratorConfig,
                 learning_engine: Optional[CompoundLearningEngine] = None,
                 integration_engine: Optional[CompoundIntegrationEngine] = None):
        super().__init__(config)
        
        # Initialize or use provided engines
        self.learning_engine = learning_engine or CompoundLearningEngine()
        self.integration_engine = integration_engine or CompoundIntegrationEngine(
            learning_engine=self.learning_engine
        )
        
        # Register compound learning extension
        self.extensions.register(
            CompoundLearningExtension(config, self.learning_engine)
        )
        
        # Track workflow execution for optimization
        self.workflow_executions: List[Dict[str, Any]] = []
        self.adaptive_decisions: List[Dict[str, Any]] = []
    
    async def run_adaptive_session(self, task: Task, 
                                  workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run orchestration with adaptive learning and workflow optimization
        """
        workflow_id = workflow_id or f"workflow_{task.id}_{int(time.time())}"
        
        # Initialize adaptive workflow
        workflow = self.learning_engine.create_adaptive_workflow(
            workflow_id=workflow_id,
            initial_models=[],  # Will be populated by actions
            input_signature=task.context
        )
        
        logger.info(f"Starting adaptive orchestration session: {workflow_id}")
        
        # Run normal session
        result = await self.run_session(task)
        
        # Extract models used in this session
        models_used = self._extract_models_from_history()
        
        if models_used:
            workflow.model_sequence = models_used
            
            # Optimize the workflow based on learning
            optimization = self.learning_engine.optimize_workflow(workflow_id)
            
            if 'optimized' in optimization and optimization['optimized']:
                logger.info(f"Workflow optimized with estimated improvement: "
                           f"{optimization['estimated_improvement']:.2%}")
                
                self.adaptive_decisions.append({
                    'workflow_id': workflow_id,
                    'optimization': optimization,
                    'timestamp': datetime.now().isoformat(),
                })
        
        # Record execution metrics
        execution_record = {
            'workflow_id': workflow_id,
            'task_id': task.id,
            'success': result.get('success', False),
            'iterations': self.iteration_count,
            'duration_seconds': result.get('elapsed_time', 0),
            'models_used': models_used,
            'timestamp': datetime.now().isoformat(),
        }
        
        self.workflow_executions.append(execution_record)
        
        return {
            **result,
            'workflow_id': workflow_id,
            'adaptive_optimizations': self.adaptive_decisions[-1] if self.adaptive_decisions else None,
        }
    
    def _extract_models_from_history(self) -> List[str]:
        """Extract model identifiers from action history"""
        models = []
        seen = set()
        
        for action in self.actions_history:
            if action.type == ActionType.INFERENCE:
                model = action.params.get('model', 'unknown')
                if model not in seen:
                    models.append(model)
                    seen.add(model)
        
        return models
    
    async def suggest_workflow_improvements(self, workflow_id: str) -> Dict[str, Any]:
        """Get suggestions for improving a workflow"""
        if workflow_id not in self.learning_engine.workflow_history:
            return {'error': f'Workflow {workflow_id} not found'}
        
        workflow = self.learning_engine.workflow_history[workflow_id]
        
        # Get complementary model suggestions
        suggestions = []
        for model in workflow.model_sequence:
            complementary = self.learning_engine.find_complementary_models(model)
            if complementary:
                suggestions.append({
                    'model': model,
                    'complementary_models': complementary,
                })
        
        # Get optimization opportunities
        optimization = self.learning_engine.optimize_workflow(workflow_id)
        
        return {
            'workflow_id': workflow_id,
            'current_sequence': workflow.model_sequence,
            'model_pairing_suggestions': suggestions,
            'optimization_opportunities': optimization,
            'timestamp': datetime.now().isoformat(),
        }
    
    def get_compound_orchestration_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics on compound learning orchestration"""
        return {
            'total_workflows': len(self.workflow_executions),
            'successful_workflows': sum(1 for w in self.workflow_executions if w['success']),
            'total_optimizations': len(self.adaptive_decisions),
            'learning_stats': self.learning_engine.get_learning_stats(),
            'integration_stats': self.integration_engine.get_integration_stats(),
            'extension_stats': self.extensions.get_stats(),
            'workflow_executions': self.workflow_executions[-10:],  # Last 10
            'timestamp': datetime.now().isoformat(),
        }


async def demonstrate_compound_orchestration():
    """Demonstration of compound learning orchestrator"""
    
    # Create configuration
    config = OrchestratorConfig(
        max_iterations=5,
        temperature=0.7,
        verbose=True,
    )
    
    # Create orchestrator with learning
    orchestrator = AdaptiveWorkflowOrchestrator(config)
    
    # Register some models in learning engine
    orchestrator.learning_engine.register_model('gpt-4', ['nlp', 'reasoning'])
    orchestrator.learning_engine.register_model('vision-model', ['vision', 'image_analysis'])
    orchestrator.learning_engine.register_model('claude', ['nlp', 'code_generation'])
    
    # Create a task
    task = Task(
        id='compound_demo_1',
        title='Analyze complex document with multiple models',
        description='Use multiple models for comprehensive analysis',
        max_iterations=3,
    )
    
    # Run adaptive session
    print("\n" + "="*70)
    print("COMPOUND LEARNING ORCHESTRATION DEMONSTRATION")
    print("="*70 + "\n")
    
    result = await orchestrator.run_adaptive_session(task)
    
    # Get statistics
    stats = orchestrator.get_compound_orchestration_stats()
    
    print("\n" + "="*70)
    print("ORCHESTRATION STATISTICS")
    print("="*70)
    print(f"Total Workflows: {stats['total_workflows']}")
    print(f"Learning Stats: {stats['learning_stats']}")
    print(f"Integration Stats: {stats['integration_stats']}")
    print("\n")
    
    return result, stats


if __name__ == "__main__":
    # Run demonstration
    result, stats = asyncio.run(demonstrate_compound_orchestration())
    print("Demonstration completed!")
