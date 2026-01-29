"""
NGVT Compound Learning System
Advanced meta-learning with knowledge accumulation and transfer
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import math

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

class CompoundLearningEngine:
    """
    Meta-learning engine that learns from interactions and improves over time
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
        }
        
        # Pattern relationships (graph)
        self.pattern_relationships: Dict[str, List[str]] = {}
        
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
            'pattern_relationships': len(self.pattern_relationships),
            'timestamp': datetime.now().isoformat(),
        }


class CompoundIntegrationEngine:
    """
    Multi-model integration with compound cross-model learning
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.integration_paths: Dict[str, List[str]] = {}  # workflow paths
        self.cross_model_cache: Dict[str, Any] = {}
        self.integration_metrics: Dict[str, Any] = {
            'total_integrations': 0,
            'successful_integrations': 0,
            'failed_integrations': 0,
            'average_path_latency': 0.0,
            'model_compatibility_matrix': {},
        }
    
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
    
    def execute_integration_path(self, path_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a compound integration workflow"""
        if path_id not in self.integration_paths:
            return {'error': f'Path {path_id} not found'}
        
        models = self.integration_paths[path_id]
        current_data = input_data
        results = []
        start_time = time.time()
        
        for model_id in models:
            if model_id not in self.models:
                return {'error': f'Model {model_id} not found'}
            
            # Execute model (simulated)
            model_result = self._execute_model(model_id, current_data)
            results.append(model_result)
            
            # Update metrics
            self.models[model_id]['call_count'] += 1
            if model_result.get('success'):
                self.models[model_id]['success_count'] += 1
            
            # Use output as input for next model
            current_data = model_result.get('output', current_data)
        
        total_latency = (time.time() - start_time) * 1000
        
        self.integration_metrics['total_integrations'] += 1
        if all(r.get('success') for r in results):
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
            'success': all(r.get('success') for r in results),
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
        """Suggest optimal integration paths based on model compatibility"""
        if len(self.models) < 2:
            return []
        
        model_ids = list(self.models.keys())
        suggestions = []
        
        # Try 2-model paths
        for i, m1 in enumerate(model_ids):
            for m2 in model_ids[i+1:]:
                compat = self.get_model_compatibility(m1, m2)
                if compat > 0.5:
                    suggestions.append({
                        'path': [m1, m2],
                        'compatibility_score': compat,
                        'path_id': f'{m1}-{m2}',
                    })
        
        # Return top suggestions
        return sorted(suggestions, key=lambda x: x['compatibility_score'], reverse=True)[:5]
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            'total_integrations': self.integration_metrics['total_integrations'],
            'successful_integrations': self.integration_metrics['successful_integrations'],
            'failed_integrations': self.integration_metrics['failed_integrations'],
            'success_rate': (
                self.integration_metrics['successful_integrations'] / 
                self.integration_metrics['total_integrations']
                if self.integration_metrics['total_integrations'] > 0 else 0.0
            ),
            'average_path_latency_ms': self.integration_metrics['average_path_latency'],
            'total_models': len(self.models),
            'total_paths': len(self.integration_paths),
            'timestamp': datetime.now().isoformat(),
        }
