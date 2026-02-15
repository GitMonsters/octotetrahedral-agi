"""
Test suite for compound learning and integration enhancements
Tests cross-model learning, workflow optimization, and orchestrator integration
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

from ngvt_compound_learning import (
    CompoundLearningEngine,
    CompoundIntegrationEngine,
    LearningExperience,
    CompoundLearningPattern,
    CrossModelLearning,
    IntegrationWorkflow,
)


class TestCompoundLearningEngine:
    """Test compound learning functionality"""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a fresh learning engine for each test"""
        return CompoundLearningEngine(max_patterns=1000)
    
    def test_model_registration(self, learning_engine):
        """Test registering models with capabilities"""
        learning_engine.register_model('gpt-4', ['nlp', 'reasoning'])
        learning_engine.register_model('vision-model', ['vision', 'image_analysis'])
        
        assert 'gpt-4' in learning_engine.model_signatures
        assert 'vision-model' in learning_engine.model_signatures
        assert 'nlp' in learning_engine.model_signatures['gpt-4']
        assert 'vision' in learning_engine.model_signatures['vision-model']
    
    def test_cross_model_transfer_recording(self, learning_engine):
        """Test recording cross-model knowledge transfer"""
        learning_engine.register_model('model_a', ['nlp'])
        learning_engine.register_model('model_b', ['nlp'])
        
        # Create a pattern
        pattern = CompoundLearningPattern(
            pattern_id='test_pattern',
            query_hash='hash123',
            response_template='template',
            accuracy=0.85,
            frequency=5,
            avg_latency_ms=50.0,
            effectiveness_score=0.8,
        )
        
        # Record transfer
        learning_engine.record_cross_model_transfer(
            pattern=pattern,
            source_model='model_a',
            target_model='model_b',
            success=True,
            effectiveness=0.9
        )
        
        assert 'model_a' in learning_engine.cross_model_transfers
        assert len(learning_engine.cross_model_transfers['model_a']) == 1
        assert learning_engine.model_affinity_matrix['model_a']['model_b'] == 0.9
    
    def test_applicable_patterns_for_model(self, learning_engine):
        """Test finding patterns applicable to a model"""
        learning_engine.register_model('model_x', ['nlp', 'translation'])
        
        # Add some experiences
        for i in range(5):
            exp = LearningExperience(
                query=f"query_{i}",
                response=f"response_{i}",
                latency_ms=50.0 + i*10,
                success=True,
                timestamp=datetime.now().isoformat(),
            )
            learning_engine.record_experience(exp)
        
        # Extract patterns
        learning_engine.extract_patterns(min_frequency=1)
        
        # Get applicable patterns
        applicable = learning_engine.get_applicable_patterns_for_model('model_x', top_k=3)
        
        # Should return patterns (may be empty initially)
        assert isinstance(applicable, list)
    
    def test_workflow_optimization(self, learning_engine):
        """Test workflow optimization based on learning"""
        learning_engine.register_model('nlp_model', ['nlp'])
        learning_engine.register_model('vision_model', ['vision'])
        learning_engine.register_model('reasoning_model', ['reasoning'])
        
        # Create workflow
        workflow = learning_engine.create_adaptive_workflow(
            workflow_id='test_workflow',
            initial_models=['nlp_model', 'vision_model', 'reasoning_model'],
            input_signature={'text': str, 'images': list}
        )
        
        assert workflow.workflow_id == 'test_workflow'
        assert workflow.model_sequence == ['nlp_model', 'vision_model', 'reasoning_model']
        
        # Optimize it
        optimization = learning_engine.optimize_workflow('test_workflow')
        
        assert 'workflow_id' in optimization
        assert optimization['workflow_id'] == 'test_workflow'


class TestCompoundIntegrationEngine:
    """Test compound integration functionality"""
    
    @pytest.fixture
    def integration_setup(self):
        """Create learning and integration engines"""
        learning_engine = CompoundLearningEngine()
        integration_engine = CompoundIntegrationEngine(learning_engine=learning_engine)
        return learning_engine, integration_engine
    
    def test_model_registration(self, integration_setup):
        """Test model registration in integration engine"""
        learning_engine, integration_engine = integration_setup
        
        integration_engine.register_model(
            'text-model',
            'nlp',
            {'max_tokens': 2000}
        )
        
        assert 'text-model' in integration_engine.models
        assert integration_engine.models['text-model']['type'] == 'nlp'
    
    def test_integration_path_definition(self, integration_setup):
        """Test defining integration paths"""
        learning_engine, integration_engine = integration_setup
        
        integration_engine.register_model('m1', 'nlp', {})
        integration_engine.register_model('m2', 'vision', {})
        
        integration_engine.define_integration_path(
            'path_1',
            ['m1', 'm2']
        )
        
        assert 'path_1' in integration_engine.integration_paths
        assert integration_engine.integration_paths['path_1'] == ['m1', 'm2']
    
    def test_integration_path_execution(self, integration_setup):
        """Test executing integration paths"""
        learning_engine, integration_engine = integration_setup
        
        integration_engine.register_model('m1', 'nlp', {})
        integration_engine.register_model('m2', 'vision', {})
        
        integration_engine.define_integration_path('path_test', ['m1', 'm2'])
        
        result = integration_engine.execute_integration_path(
            'path_test',
            {'input': 'test_data'},
            record_learning=False
        )
        
        assert result['success']
        assert result['path_id'] == 'path_test'
        assert 'total_latency_ms' in result
        assert 'final_output' in result
    
    def test_integration_with_learning_recording(self, integration_setup):
        """Test integration execution with learning recording"""
        learning_engine, integration_engine = integration_setup
        
        learning_engine.register_model('m1', ['nlp'])
        learning_engine.register_model('m2', ['vision'])
        
        integration_engine.register_model('m1', 'nlp', {})
        integration_engine.register_model('m2', 'vision', {})
        integration_engine.define_integration_path('path_with_learning', ['m1', 'm2'])
        
        # Execute with learning
        result = integration_engine.execute_integration_path(
            'path_with_learning',
            {'data': 'test'},
            record_learning=True
        )
        
        assert result['success']
        assert integration_engine.integration_metrics['cross_model_learning_events'] > 0
    
    def test_model_compatibility(self, integration_setup):
        """Test model compatibility calculation"""
        learning_engine, integration_engine = integration_setup
        
        integration_engine.register_model('m1', 'nlp', {})
        integration_engine.register_model('m2', 'nlp', {})
        integration_engine.register_model('m3', 'vision', {})
        
        # Same type should have higher compatibility
        compat_same = integration_engine.get_model_compatibility('m1', 'm2')
        compat_diff = integration_engine.get_model_compatibility('m1', 'm3')
        
        assert compat_same == 0.8
        assert compat_diff > 0  # Should have some compatibility
    
    def test_integration_suggestions(self, integration_setup):
        """Test getting integration path suggestions"""
        learning_engine, integration_engine = integration_setup
        
        integration_engine.register_model('nlp1', 'nlp', {})
        integration_engine.register_model('nlp2', 'nlp', {})
        integration_engine.register_model('vision1', 'vision', {})
        
        suggestions = integration_engine.suggest_integration_paths()
        
        assert isinstance(suggestions, list)
        # Should suggest paths with good compatibility
        if suggestions:
            assert 'path' in suggestions[0]
            assert 'compatibility_score' in suggestions[0]


class TestCrossModelLearning:
    """Test cross-model learning mechanisms"""
    
    def test_complementary_models_discovery(self):
        """Test finding models that work well together"""
        learning_engine = CompoundLearningEngine()
        
        learning_engine.register_model('nlp_a', ['nlp', 'text'])
        learning_engine.register_model('vision_b', ['vision', 'images'])
        
        # Record successful transfers
        pattern = CompoundLearningPattern(
            pattern_id='p1',
            query_hash='h1',
            response_template='t1',
            accuracy=0.9,
            frequency=10,
            avg_latency_ms=50.0,
            effectiveness_score=0.85,
        )
        
        learning_engine.record_cross_model_transfer(
            pattern, 'nlp_a', 'vision_b', True, 0.95
        )
        
        # Find complementary models
        complementary = learning_engine.find_complementary_models('nlp_a')
        
        assert 'vision_b' in complementary
        assert complementary['vision_b'] == 0.95
    
    def test_transfer_efficiency_calculation(self):
        """Test transfer efficiency metrics"""
        learning_engine = CompoundLearningEngine()
        
        learning_engine.register_model('m1', ['type_a'])
        learning_engine.register_model('m2', ['type_a'])
        
        # Record experiences
        for i in range(5):
            exp = LearningExperience(
                query=f"q{i}",
                response=f"r{i}",
                latency_ms=40.0 + i*5,
                success=i < 4,  # 4 successes, 1 failure
                timestamp=datetime.now().isoformat(),
                metadata={'model': 'm1'},
            )
            learning_engine.record_experience(exp)
        
        learning_engine.extract_patterns(min_frequency=2)
        cycle = learning_engine.compound_learning_cycle()
        
        assert 'transfer_efficiency' in cycle
        assert 'cumulative_accuracy' in cycle
        assert cycle['cumulative_accuracy'] == 0.8  # 4/5 successes


class TestAdaptiveWorkflow:
    """Test adaptive workflow functionality"""
    
    def test_workflow_creation(self):
        """Test creating adaptive workflows"""
        learning_engine = CompoundLearningEngine()
        
        workflow = learning_engine.create_adaptive_workflow(
            workflow_id='adaptive_1',
            initial_models=['m1', 'm2', 'm3'],
            input_signature={'input': 'data'}
        )
        
        assert workflow.workflow_id == 'adaptive_1'
        assert workflow.model_sequence == ['m1', 'm2', 'm3']
        assert workflow.execution_count == 0
    
    def test_workflow_sequence_optimization(self):
        """Test optimizing model sequences"""
        learning_engine = CompoundLearningEngine()
        
        # Register models
        for m in ['m1', 'm2', 'm3']:
            learning_engine.register_model(m, ['general'])
        
        # Set up affinities (m1 works best with m2, m2 works best with m3)
        learning_engine.model_affinity_matrix['m1']['m2'] = 0.95
        learning_engine.model_affinity_matrix['m2']['m3'] = 0.95
        learning_engine.model_affinity_matrix['m1']['m3'] = 0.5
        learning_engine.model_affinity_matrix['m2']['m1'] = 0.90
        learning_engine.model_affinity_matrix['m3']['m1'] = 0.5
        learning_engine.model_affinity_matrix['m3']['m2'] = 0.90
        
        # Create workflow
        workflow = learning_engine.create_adaptive_workflow(
            workflow_id='opt_test',
            initial_models=['m1', 'm3', 'm2'],
            input_signature={}
        )
        
        # Optimize
        optimization = learning_engine.optimize_workflow('opt_test')
        
        assert 'optimized_sequence' in optimization
        # Optimized should be ['m1', 'm2', 'm3'] due to affinities


def run_compound_integration_demo():
    """Demo showing compound learning integration in action"""
    print("\n" + "="*70)
    print("COMPOUND INTEGRATION DEMONSTRATION")
    print("="*70 + "\n")
    
    # Initialize engines
    learning_engine = CompoundLearningEngine()
    integration_engine = CompoundIntegrationEngine(learning_engine=learning_engine)
    
    # Register models
    models = [
        ('text-analyzer', 'nlp', ['text analysis', 'sentiment']),
        ('image-processor', 'vision', ['image classification', 'detection']),
        ('code-generator', 'nlp', ['code generation', 'refactoring']),
        ('summarizer', 'nlp', ['summarization', 'extraction']),
    ]
    
    for model_id, model_type, capabilities in models:
        learning_engine.register_model(model_id, capabilities)
        integration_engine.register_model(model_id, model_type, {})
    
    # Define integration paths
    paths = [
        ('text_analysis_pipeline', ['text-analyzer', 'summarizer']),
        ('multimodal_pipeline', ['image-processor', 'text-analyzer']),
        ('code_workflow', ['code-generator', 'text-analyzer']),
    ]
    
    for path_id, model_sequence in paths:
        integration_engine.define_integration_path(path_id, model_sequence)
    
    print("Registered Models:")
    for model_id in learning_engine.model_signatures.keys():
        caps = learning_engine.model_signatures[model_id]
        print(f"  - {model_id}: {caps}")
    
    print("\nDefined Integration Paths:")
    for path_id, models in integration_engine.integration_paths.items():
        print(f"  - {path_id}: {' -> '.join(models)}")
    
    # Execute paths with learning
    print("\nExecuting integration paths with learning...")
    for path_id in integration_engine.integration_paths.keys():
        result = integration_engine.execute_integration_path(
            path_id,
            {'sample': 'input_data'},
            record_learning=True
        )
        print(f"  {path_id}: {'✓' if result['success'] else '✗'} "
              f"({result['total_latency_ms']:.1f}ms)")
    
    # Get optimization suggestions
    print("\nIntegration Path Suggestions:")
    suggestions = integration_engine.suggest_integration_paths()
    for i, sugg in enumerate(suggestions, 1):
        print(f"  {i}. {' -> '.join(sugg['path'])}")
        print(f"     Compatibility: {sugg['compatibility_score']:.2f}, "
              f"Learning Boost: {sugg['learning_boost']:.2%}")
    
    # Display stats
    print("\nCompound Integration Statistics:")
    stats = integration_engine.get_integration_stats()
    print(f"  Total Integrations: {stats['total_integrations']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Cross-Model Learning Events: {stats['cross_model_learning_events']}")
    
    learning_stats = learning_engine.get_learning_stats()
    print(f"\nCompound Learning Statistics:")
    print(f"  Total Experiences: {learning_stats['total_experiences']}")
    print(f"  Transfer Efficiency: {learning_stats['transfer_efficiency']:.2%}")
    print(f"  Cross-Model Efficiency: {learning_stats['cross_model_efficiency']:.2%}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Run demonstration
    run_compound_integration_demo()
    
    # Run pytest if available
    try:
        pytest.main([__file__, '-v'])
    except:
        print("Install pytest to run full test suite: pip install pytest")
