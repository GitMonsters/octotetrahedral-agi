"""
NGVT Compound Learning System
Advanced meta-learning with knowledge accumulation and transfer
Supports cross-model learning, orchestrator integration, and adaptive compound workflows
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import math
import asyncio

@dataclass
class LearningExperience:
    """Single learning experience/interaction"""
    query: str
    response: str
    latency_ms: float
    success: bool
    timestamp: str
    cache_hit: bool = False
    tokens_generated: int = 0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompoundLearningPattern:
    """Pattern learned from compound experiences"""
    pattern_id: str
    query_hash: str
    response_template: str
    accuracy: float  # 0-1, based on success history
    frequency: int  # how many times this pattern appears
    avg_latency_ms: float
    optimal_parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    last_used: str = ""
    effectiveness_score: float = 0.0
    model_sources: List[str] = field(default_factory=list)  # Which models generated this pattern
    cross_model_applicability: Dict[str, float] = field(default_factory=dict)  # Model -> applicability score

@dataclass
class CrossModelLearning:
    """Learning transferred between models in compound workflows"""
    source_model: str
    target_model: str
    pattern_id: str
    transfer_score: float  # 0-1, how well pattern transferred
    successful_applications: int = 0
    failed_applications: int = 0
    created_at: str = ""
    last_applied: str = ""

@dataclass
class IntegrationWorkflow:
    """Represents a compound integration workflow with learning"""
    workflow_id: str
    model_sequence: List[str]
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    execution_count: int = 0
    learned_optimizations: List[str] = field(default_factory=list)  # Applied patterns
    created_at: str = ""
    last_executed: str = ""
    effectiveness_trend: List[float] = field(default_factory=list)  # Historical effectiveness

class CompoundLearningEngine:
    """
    Meta-learning engine that learns from interactions and improves over time
    Supports cross-model learning and compound workflow optimization
    """
    
    def __init__(self, max_patterns: int = 10000, learning_rate: float = 0.01):
        self.max_patterns = max_patterns
        self.learning_rate = learning_rate
        
        # Core learning structures
        self.experiences: List[LearningExperience] = []
        self.patterns: Dict[str, CompoundLearningPattern] = {}
        self.query_vectors: Dict[str, List[float]] = {}  # Embeddings
        
        # Compound metrics
        self.knowledge_base: Dict[str, Any] = {
            'total_experiences': 0,
            'total_patterns': 0,
            'total_learning_cycles': 0,
            'cumulative_accuracy': 0.0,
            'knowledge_density': 0.0,
            'transfer_efficiency': 0.0,
            'cross_model_efficiency': 0.0,
        }
        
        # Pattern relationships (graph)
        self.pattern_relationships: Dict[str, List[str]] = {}
        
        # Cross-model learning
        self.cross_model_transfers: Dict[str, List[CrossModelLearning]] = {}  # source_model -> transfers
        self.model_signatures: Dict[str, Set[str]] = {}  # model -> capability signatures
        self.workflow_history: Dict[str, IntegrationWorkflow] = {}  # workflow_id -> workflow
        self.model_affinity_matrix: Dict[str, Dict[str, float]] = {}  # model1 -> {model2 -> affinity}
        
    def record_experience(self, experience: LearningExperience) -> None:
        """Record a new learning experience"""
        self.experiences.append(experience)
        self.knowledge_base['total_experiences'] += 1
        
        # Update knowledge density
        self._update_knowledge_density()
    
    def extract_patterns(self, min_frequency: int = 3) -> List[CompoundLearningPattern]:
        """
        Extract recurring patterns from experiences using compound analysis
        """
        pattern_counts: Dict[str, List[LearningExperience]] = {}
        
        # Group similar experiences
        for exp in self.experiences:
            query_hash = self._hash_query(exp.query)
            if query_hash not in pattern_counts:
                pattern_counts[query_hash] = []
            pattern_counts[query_hash].append(exp)
        
        # Create patterns from frequent groups
        new_patterns = []
        for query_hash, exps in pattern_counts.items():
            if len(exps) >= min_frequency:
                pattern = self._create_pattern_from_experiences(query_hash, exps)
                self.patterns[pattern.pattern_id] = pattern
                new_patterns.append(pattern)
        
        self.knowledge_base['total_patterns'] = len(self.patterns)
        return new_patterns
    
    def _create_pattern_from_experiences(self, query_hash: str, 
                                        experiences: List[LearningExperience]
                                        ) -> CompoundLearningPattern:
        """Create a pattern from a group of similar experiences"""
        # Calculate statistics
        successes = sum(1 for e in experiences if e.success)
        accuracy = successes / len(experiences) if experiences else 0.0
        avg_latency = sum(e.latency_ms for e in experiences) / len(experiences)
        
        # Extract optimal parameters
        optimal_params = self._extract_parameters(experiences)
        
        # Create pattern ID
        pattern_id = f"pattern_{query_hash[:12]}_{len(self.patterns)}"
        
        # Calculate effectiveness score
        effectiveness = accuracy * (1 - min(avg_latency / 1000, 1.0))
        
        pattern = CompoundLearningPattern(
            pattern_id=pattern_id,
            query_hash=query_hash,
            response_template=experiences[0].response[:100] + "...",
            accuracy=accuracy,
            frequency=len(experiences),
            avg_latency_ms=avg_latency,
            optimal_parameters=optimal_params,
            created_at=datetime.now().isoformat(),
            effectiveness_score=effectiveness
        )
        
        return pattern
    
    def _extract_parameters(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Extract optimal parameters from successful experiences"""
        successful = [e for e in experiences if e.success]
        
        if not successful:
            return {}
        
        return {
            'avg_tokens': sum(e.tokens_generated for e in successful) / len(successful),
            'avg_latency': sum(e.latency_ms for e in successful) / len(successful),
            'cache_hit_rate': sum(1 for e in successful if e.cache_hit) / len(successful),
            'avg_confidence': sum(e.confidence for e in successful) / len(successful),
        }
    
    def predict_with_learning(self, query: str) -> Dict[str, Any]:
        """
        Use learned patterns to predict optimal response
        """
        query_hash = self._hash_query(query)
        
        # Check if we have learned about this query
        if query_hash in self.patterns:
            pattern = self.patterns[query_hash]
            prediction = {
                'has_learned_pattern': True,
                'pattern_id': pattern.pattern_id,
                'predicted_accuracy': pattern.accuracy,
                'predicted_latency_ms': pattern.avg_latency_ms,
                'effectiveness_score': pattern.effectiveness_score,
                'optimal_parameters': pattern.optimal_parameters,
                'confidence': min(0.5 + (pattern.frequency / 100), 1.0),  # Increases with frequency
            }
            return prediction
        
        # Check for similar patterns (transfer learning)
        similar_patterns = self._find_similar_patterns(query)
        if similar_patterns:
            best_match = similar_patterns[0]
            prediction = {
                'has_learned_pattern': False,
                'similar_pattern_found': True,
                'pattern_id': best_match['pattern'].pattern_id,
                'similarity_score': best_match['similarity'],
                'transferred_parameters': best_match['pattern'].optimal_parameters,
                'confidence': best_match['similarity'] * 0.8,  # Transfer learning confidence
            }
            return prediction
        
        return {
            'has_learned_pattern': False,
            'similar_pattern_found': False,
            'confidence': 0.0,
        }
    
    def _find_similar_patterns(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar patterns using similarity matching"""
        similarities = []
        
        for pattern in self.patterns.values():
            # Simple similarity: based on effectiveness and frequency
            similarity = pattern.effectiveness_score * (1 + math.log(pattern.frequency + 1) / 10)
            similarities.append({
                'pattern': pattern,
                'similarity': min(similarity, 1.0),
            })
        
        # Return top K similar patterns
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def compound_learning_cycle(self) -> Dict[str, Any]:
        """
        Run a compound learning cycle to improve meta-knowledge
        """
        self.knowledge_base['total_learning_cycles'] += 1
        
        # Extract new patterns
        new_patterns = self.extract_patterns(min_frequency=2)
        
        # Build pattern relationships
        self._build_pattern_relationships()
        
        # Calculate transfer efficiency
        transfer_efficiency = self._calculate_transfer_efficiency()
        self.knowledge_base['transfer_efficiency'] = transfer_efficiency
        
        # Calculate cumulative accuracy
        if self.experiences:
            cumulative_accuracy = sum(1 for e in self.experiences if e.success) / len(self.experiences)
            self.knowledge_base['cumulative_accuracy'] = cumulative_accuracy
        
        return {
            'cycle_number': self.knowledge_base['total_learning_cycles'],
            'patterns_discovered': len(new_patterns),
            'total_patterns': len(self.patterns),
            'transfer_efficiency': transfer_efficiency,
            'cumulative_accuracy': self.knowledge_base['cumulative_accuracy'],
            'timestamp': datetime.now().isoformat(),
        }
    
    def _build_pattern_relationships(self) -> None:
        """Build a graph of pattern relationships"""
        self.pattern_relationships = {}
        
        patterns_list = list(self.patterns.values())
        for i, pattern1 in enumerate(patterns_list):
            related = []
            for pattern2 in patterns_list[i+1:]:
                # Simple relationship: patterns with similar effectiveness
                if abs(pattern1.effectiveness_score - pattern2.effectiveness_score) < 0.1:
                    related.append(pattern2.pattern_id)
            
            if related:
                self.pattern_relationships[pattern1.pattern_id] = related
    
    def _calculate_transfer_efficiency(self) -> float:
        """Calculate how effectively we're transferring knowledge"""
        if not self.patterns:
            return 0.0
        
        # Average effectiveness of all patterns
        avg_effectiveness = sum(p.effectiveness_score for p in self.patterns.values()) / len(self.patterns)
        
        # Average relationships (knowledge connectivity)
        avg_relationships = sum(len(v) for v in self.pattern_relationships.values()) / len(self.patterns)
        
        # Combined transfer efficiency
        transfer_efficiency = (avg_effectiveness + min(avg_relationships / 5, 1.0)) / 2
        
        return min(transfer_efficiency, 1.0)
    
    def _update_knowledge_density(self) -> None:
        """Update knowledge density metric"""
        if not self.experiences or not self.patterns:
            self.knowledge_base['knowledge_density'] = 0.0
            return
        
        # Knowledge density = patterns / experiences (compression ratio)
        density = len(self.patterns) / len(self.experiences) if self.experiences else 0.0
        self.knowledge_base['knowledge_density'] = min(density, 1.0)
    
    def _hash_query(self, query: str) -> str:
        """Create a hash of the query for grouping"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        return {
            'total_experiences': self.knowledge_base['total_experiences'],
            'total_patterns': self.knowledge_base['total_patterns'],
            'learning_cycles': self.knowledge_base['total_learning_cycles'],
            'cumulative_accuracy': self.knowledge_base['cumulative_accuracy'],
            'knowledge_density': self.knowledge_base['knowledge_density'],
            'transfer_efficiency': self.knowledge_base['transfer_efficiency'],
            'cross_model_efficiency': self.knowledge_base['cross_model_efficiency'],
            'pattern_relationships': len(self.pattern_relationships),
            'active_models': len(self.model_signatures),
            'workflows': len(self.workflow_history),
            'timestamp': datetime.now().isoformat(),
        }
    
    # ============ CROSS-MODEL LEARNING METHODS ============
    
    def register_model(self, model_id: str, capabilities: List[str]) -> None:
        """Register a model with its capabilities"""
        self.model_signatures[model_id] = set(capabilities)
        self.model_affinity_matrix[model_id] = {}
    
    def record_cross_model_transfer(self, pattern: CompoundLearningPattern, 
                                   source_model: str, target_model: str,
                                   success: bool, effectiveness: float) -> None:
        """Record knowledge transfer between models"""
        if source_model not in self.cross_model_transfers:
            self.cross_model_transfers[source_model] = []
        
        transfer = CrossModelLearning(
            source_model=source_model,
            target_model=target_model,
            pattern_id=pattern.pattern_id,
            transfer_score=effectiveness,
            successful_applications=1 if success else 0,
            failed_applications=0 if success else 1,
            created_at=datetime.now().isoformat(),
            last_applied=datetime.now().isoformat(),
        )
        
        self.cross_model_transfers[source_model].append(transfer)
        
        # Update model affinity
        if target_model not in self.model_affinity_matrix[source_model]:
            self.model_affinity_matrix[source_model][target_model] = effectiveness
        else:
            # Exponential moving average
            current = self.model_affinity_matrix[source_model][target_model]
            self.model_affinity_matrix[source_model][target_model] = (
                current * 0.7 + effectiveness * 0.3
            )
        
        # Update pattern applicability
        if target_model not in pattern.cross_model_applicability:
            pattern.cross_model_applicability[target_model] = effectiveness
        else:
            pattern.cross_model_applicability[target_model] = (
                pattern.cross_model_applicability[target_model] * 0.7 + effectiveness * 0.3
            )
    
    def get_applicable_patterns_for_model(self, model_id: str, 
                                         top_k: int = 5) -> List[CompoundLearningPattern]:
        """Get patterns most applicable to a specific model"""
        applicable = []
        
        for pattern in self.patterns.values():
            # Score based on model applicability and effectiveness
            score = pattern.cross_model_applicability.get(model_id, 0.0)
            score *= pattern.effectiveness_score  # Favor effective patterns
            
            applicable.append((pattern, score))
        
        # Sort by score and return top K
        applicable.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in applicable[:top_k]]
    
    def find_complementary_models(self, model_id: str) -> Dict[str, float]:
        """Find models that work well with the given model (based on learning history)"""
        if model_id not in self.model_affinity_matrix:
            return {}
        
        # Return affinity scores
        affinities = self.model_affinity_matrix[model_id]
        sorted_affinities = sorted(affinities.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_affinities)
    
    def create_adaptive_workflow(self, workflow_id: str, initial_models: List[str],
                               input_signature: Dict[str, Any]) -> IntegrationWorkflow:
        """Create a workflow that learns and adapts based on execution"""
        workflow = IntegrationWorkflow(
            workflow_id=workflow_id,
            model_sequence=initial_models,
            created_at=datetime.now().isoformat(),
        )
        
        self.workflow_history[workflow_id] = workflow
        return workflow
    
    def optimize_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Optimize a workflow based on learned patterns and model affinities"""
        if workflow_id not in self.workflow_history:
            return {'error': f'Workflow {workflow_id} not found'}
        
        workflow = self.workflow_history[workflow_id]
        original_sequence = workflow.model_sequence.copy()
        
        # Try to reorder models based on affinity scores
        if len(workflow.model_sequence) > 1:
            optimized = self._optimize_model_sequence(workflow.model_sequence)
            workflow.model_sequence = optimized
        
        # Find applicable patterns to apply
        applicable_patterns = []
        for model in workflow.model_sequence:
            patterns = self.get_applicable_patterns_for_model(model, top_k=3)
            applicable_patterns.extend([p.pattern_id for p in patterns])
        
        workflow.learned_optimizations = list(set(applicable_patterns))
        workflow.last_executed = datetime.now().isoformat()
        
        return {
            'workflow_id': workflow_id,
            'optimized': original_sequence != workflow.model_sequence,
            'original_sequence': original_sequence,
            'optimized_sequence': workflow.model_sequence,
            'applied_patterns': workflow.learned_optimizations,
            'estimated_improvement': self._estimate_improvement(original_sequence, 
                                                               workflow.model_sequence),
        }
    
    def _optimize_model_sequence(self, models: List[str]) -> List[str]:
        """Reorder models for better workflow efficiency"""
        if len(models) <= 1:
            return models
        
        # Start with first model
        optimized = [models[0]]
        remaining = set(models[1:])
        
        while remaining:
            current = optimized[-1]
            # Find best next model based on affinity
            best_next = None
            best_score = -1
            
            for candidate in remaining:
                score = self.model_affinity_matrix.get(current, {}).get(candidate, 0.3)
                if score > best_score:
                    best_score = score
                    best_next = candidate
            
            if best_next:
                optimized.append(best_next)
                remaining.remove(best_next)
            else:
                # Just pick first remaining if no affinity data
                best_next = remaining.pop()
                optimized.append(best_next)
        
        return optimized
    
    def _estimate_improvement(self, original: List[str], optimized: List[str]) -> float:
        """Estimate performance improvement from sequence reordering"""
        # Calculate average affinity for each sequence
        def calculate_sequence_score(sequence: List[str]) -> float:
            if len(sequence) <= 1:
                return 1.0
            
            total_affinity = 0.0
            for i in range(len(sequence) - 1):
                affinity = self.model_affinity_matrix.get(sequence[i], {}).get(sequence[i+1], 0.5)
                total_affinity += affinity
            
            return total_affinity / (len(sequence) - 1)
        
        original_score = calculate_sequence_score(original)
        optimized_score = calculate_sequence_score(optimized)
        
        if original_score == 0:
            return 0.0
        
        return (optimized_score - original_score) / original_score


class CompoundIntegrationEngine:
    """
    Multi-model integration with compound cross-model learning
    Coordinates workflows between models and learns optimal integration patterns
    """
    
    def __init__(self, learning_engine: Optional['CompoundLearningEngine'] = None):
        self.models: Dict[str, Any] = {}
        self.integration_paths: Dict[str, List[str]] = {}  # workflow paths
        self.cross_model_cache: Dict[str, Any] = {}
        self.integration_metrics: Dict[str, Any] = {
            'total_integrations': 0,
            'successful_integrations': 0,
            'failed_integrations': 0,
            'average_path_latency': 0.0,
            'model_compatibility_matrix': {},
            'cross_model_learning_events': 0,
            'optimization_improvements': 0.0,
        }
        
        # Link to learning engine for knowledge transfer
        self.learning_engine = learning_engine
    
    def register_model(self, model_id: str, model_type: str, config: Dict[str, Any]) -> None:
        """Register a new model for integration"""
        self.models[model_id] = {
            'type': model_type,
            'config': config,
            'registered_at': datetime.now().isoformat(),
            'call_count': 0,
            'success_count': 0,
        }
    
    def define_integration_path(self, path_id: str, model_sequence: List[str]) -> None:
        """Define a compound integration path (workflow)"""
        self.integration_paths[path_id] = model_sequence
    
    def execute_integration_path(self, path_id: str, input_data: Dict[str, Any],
                               record_learning: bool = True) -> Dict[str, Any]:
        """Execute a compound integration workflow with optional learning"""
        if path_id not in self.integration_paths:
            return {'error': f'Path {path_id} not found'}
        
        models = self.integration_paths[path_id]
        current_data = input_data
        results = []
        start_time = time.time()
        path_success = True
        
        for i, model_id in enumerate(models):
            if model_id not in self.models:
                return {'error': f'Model {model_id} not found'}
            
            # Execute model (simulated)
            model_result = self._execute_model(model_id, current_data)
            results.append(model_result)
            
            # Update metrics
            self.models[model_id]['call_count'] += 1
            if model_result.get('success'):
                self.models[model_id]['success_count'] += 1
            else:
                path_success = False
            
            # Record cross-model transfer if learning engine available
            if record_learning and self.learning_engine and i > 0:
                prev_model = models[i-1]
                transfer_effectiveness = 0.9 if model_result.get('success') else 0.3
                self._record_model_transition_learning(prev_model, model_id, transfer_effectiveness)
            
            # Use output as input for next model
            current_data = model_result.get('output', current_data)
        
        total_latency = (time.time() - start_time) * 1000
        
        self.integration_metrics['total_integrations'] += 1
        if path_success and all(r.get('success') for r in results):
            self.integration_metrics['successful_integrations'] += 1
        else:
            self.integration_metrics['failed_integrations'] += 1
        
        # Update average path latency
        current_avg = self.integration_metrics.get('average_path_latency', 0.0)
        count = self.integration_metrics['total_integrations']
        new_avg = (current_avg * (count - 1) + total_latency) / count
        self.integration_metrics['average_path_latency'] = new_avg
        
        return {
            'path_id': path_id,
            'success': path_success,
            'results': results,
            'total_latency_ms': total_latency,
            'final_output': current_data,
        }
    
    def _execute_model(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single model (simulated)"""
        # Simulate model execution
        time.sleep(0.01)  # 10ms per model
        
        return {
            'model_id': model_id,
            'success': True,
            'output': {
                **input_data,
                f'{model_id}_processed': True,
                f'{model_id}_timestamp': datetime.now().isoformat(),
            },
            'latency_ms': 10.0,
        }
    
    def _record_model_transition_learning(self, source_model: str, target_model: str,
                                         effectiveness: float) -> None:
        """Record learning from model-to-model transitions"""
        if self.learning_engine:
            # Create a synthetic pattern representing the transition
            pattern = CompoundLearningPattern(
                pattern_id=f"{source_model}->{target_model}",
                query_hash=hashlib.md5(f"{source_model}-{target_model}".encode()).hexdigest(),
                response_template=f"Transfer from {source_model} to {target_model}",
                accuracy=effectiveness,
                frequency=1,
                avg_latency_ms=10.0,
                effectiveness_score=effectiveness,
                model_sources=[source_model],
            )
            
            # Record transfer
            self.learning_engine.record_cross_model_transfer(
                pattern=pattern,
                source_model=source_model,
                target_model=target_model,
                success=effectiveness > 0.7,
                effectiveness=effectiveness
            )
            
            self.integration_metrics['cross_model_learning_events'] += 1
    
    def get_model_compatibility(self, model_id1: str, model_id2: str) -> float:
        """Calculate compatibility between two models"""
        if model_id1 not in self.models or model_id2 not in self.models:
            return 0.0
        
        m1 = self.models[model_id1]
        m2 = self.models[model_id2]
        
        # Simple compatibility: same type or complementary types
        if m1['type'] == m2['type']:
            return 0.8
        
        # Check if types are complementary
        complementary = {
            'nlp': ['vision', 'speech'],
            'vision': ['nlp', 'audio'],
            'speech': ['nlp', 'audio'],
        }
        
        if m1['type'] in complementary and m2['type'] in complementary[m1['type']]:
            return 0.9
        
        return 0.5
    
    def suggest_integration_paths(self) -> List[Dict[str, Any]]:
        """Suggest optimal integration paths based on model compatibility and learning"""
        if len(self.models) < 2:
            return []
        
        model_ids = list(self.models.keys())
        suggestions = []
        
        # Try 2-model paths
        for i, m1 in enumerate(model_ids):
            for m2 in model_ids[i+1:]:
                compat = self.get_model_compatibility(m1, m2)
                
                # Get learning efficiency if available
                learning_boost = 1.0
                if self.learning_engine:
                    affinity = self.learning_engine.model_affinity_matrix.get(m1, {}).get(m2, 0.0)
                    if affinity > 0:
                        learning_boost = 1.0 + affinity * 0.2  # 20% boost from learning
                
                final_score = compat * learning_boost
                
                suggestions.append({
                    'path': [m1, m2],
                    'compatibility_score': compat,
                    'learning_boost': learning_boost - 1.0,
                    'final_score': final_score,
                    'path_id': f'{m1}-{m2}',
                })
        
        # Return top suggestions sorted by final score
        return sorted(suggestions, key=lambda x: x['final_score'], reverse=True)[:5]
    
    def optimize_existing_path(self, path_id: str) -> Dict[str, Any]:
        """Use learning engine to optimize an existing integration path"""
        if not self.learning_engine or path_id not in self.integration_paths:
            return {'error': 'Cannot optimize path'}
        
        result = self.learning_engine.optimize_workflow(path_id)
        
        if 'error' not in result:
            # Update the path in integration engine
            self.integration_paths[path_id] = result.get('optimized_sequence', 
                                                        self.integration_paths[path_id])
            self.integration_metrics['optimization_improvements'] += result.get('estimated_improvement', 0)
        
        return result
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics including cross-model learning"""
        success_rate = (
            self.integration_metrics['successful_integrations'] / 
            self.integration_metrics['total_integrations']
            if self.integration_metrics['total_integrations'] > 0 else 0.0
        )
        
        return {
            'total_integrations': self.integration_metrics['total_integrations'],
            'successful_integrations': self.integration_metrics['successful_integrations'],
            'failed_integrations': self.integration_metrics['failed_integrations'],
            'success_rate': success_rate,
            'average_path_latency_ms': self.integration_metrics['average_path_latency'],
            'total_models': len(self.models),
            'total_paths': len(self.integration_paths),
            'cross_model_learning_events': self.integration_metrics.get('cross_model_learning_events', 0),
            'optimization_improvements': self.integration_metrics.get('optimization_improvements', 0.0),
            'timestamp': datetime.now().isoformat(),
        }
