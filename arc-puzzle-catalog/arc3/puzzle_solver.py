"""
RE-ARC Puzzle Solver — Integrated solver combining enhanced perception,
rule inference, and rule application.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger("arc3.puzzle_solver")


class REARCPuzzleSolver:
    """Solves RE-ARC challenges using inferred transformation rules."""

    def __init__(self, use_perception: bool = True, use_reasoning: bool = True):
        self.use_perception = use_perception
        self.use_reasoning = use_reasoning
        self.perception = None
        self.reasoning = None
        self.applicator = None
        
        if use_perception:
            try:
                from arc3.perception import PerceptionModule
                self.perception = PerceptionModule()
            except ImportError:
                logger.warning("Could not import PerceptionModule")
        
        if use_reasoning:
            try:
                from arc3.reasoning import RuleInferenceEngine
                from arc3.rule_application import RuleApplicator
                self.reasoning = RuleInferenceEngine()
                self.applicator = RuleApplicator(confidence_threshold=0.5)
            except ImportError:
                logger.warning("Could not import RuleInferenceEngine or RuleApplicator")

    def solve_task(self, task: dict) -> Optional[np.ndarray]:
        """Solve a single RE-ARC task.
        
        Args:
            task: Dict with 'train' (list of {'input', 'output'}) and 'test' (list of {'input'})
        
        Returns:
            Predicted output grid or None
        """
        if not task.get('train') or not task.get('test'):
            return None
        
        try:
            training_examples = task['train']
            test_input = np.array(task['test'][0]['input'], dtype=int)
            
            # Infer rules from training examples
            rules = self.reasoning.infer_transformation_rules(training_examples)
            
            if not rules:
                return self._fallback_heuristic(test_input, training_examples)
            
            # Load rules into applicator
            self.applicator.load_rules(rules)
            
            # Apply best rule
            prediction = self.applicator.apply_with_fallback(test_input)
            
            if prediction:
                return prediction.grid
            
            # Fallback
            return self._fallback_heuristic(test_input, training_examples)
        
        except Exception as e:
            logger.debug(f"Solver error: {e}")
            return self._fallback_heuristic(test_input, training_examples)

    def _fallback_heuristic(self, test_input: np.ndarray, 
                            training_examples: list[dict]) -> Optional[np.ndarray]:
        """Fallback: copy first training example output."""
        if training_examples:
            return np.array(training_examples[0]['output'], dtype=int)
        return None

    def solve_batch(self, tasks: list[dict], verbose: bool = False) -> list[dict]:
        """Solve a batch of tasks.
        
        Returns:
            List of results with task_id, predicted_output, confidence
        """
        results = []
        
        for i, task in enumerate(tasks):
            task_id = task.get('task_id', f'task_{i}')
            try:
                output = self.solve_task(task)
                results.append({
                    'task_id': task_id,
                    'predicted_output': output.tolist() if output is not None else None,
                    'success': output is not None,
                })
                
                if verbose:
                    status = "✓" if output is not None else "✗"
                    print(f"{status} {task_id}")
            
            except Exception as e:
                if verbose:
                    print(f"✗ {task_id}: {str(e)[:50]}")
                results.append({
                    'task_id': task_id,
                    'predicted_output': None,
                    'success': False,
                })
        
        return results

    def analyze_task(self, task: dict) -> dict:
        """Analyze a task and return detailed information."""
        if not task.get('train'):
            return {'error': 'No training examples'}
        
        training = task['train']
        
        # Analyze perception features if available
        perception_features = None
        if self.perception:
            perception_features = {}
            for i, ex in enumerate(training):
                inp = np.array(ex['input'], dtype=int)
                out = np.array(ex['output'], dtype=int)
                perception_features[f'example_{i}'] = {
                    'input_shape': inp.shape,
                    'output_shape': out.shape,
                    'shapes_input': self.perception.detect_shapes(inp),
                    'symmetry_input': self.perception.detect_symmetry(inp),
                    'colors_input': self.perception.cluster_colors(inp),
                    'connectivity_input': self.perception.measure_connectivity(inp),
                }
        
        # Analyze rules
        rules = self.reasoning.infer_transformation_rules(training) if self.reasoning else []
        
        return {
            'task_id': task.get('task_id'),
            'training_count': len(training),
            'test_count': len(task.get('test', [])),
            'perception_features': perception_features,
            'inferred_rules': [
                {
                    'type': r.rule_type,
                    'description': r.description,
                    'confidence': float(r.confidence),
                    'parameters': r.parameters,
                }
                for r in rules
            ],
        }

    def reset(self):
        """Reset solver state."""
        if self.perception:
            self.perception.reset()
        if self.reasoning:
            self.reasoning.reset()
        if self.applicator:
            self.applicator.reset()
