#!/usr/bin/env python3
"""
ARC Hybrid Solver - Combining DSL + Neural Approaches
=====================================================

Strategy:
1. Try HypothesisEngine first (few-shot + composition — solves 8%+ of tasks)
2. Try DSL-based program synthesis (fast, interpretable)
3. For unsolved tasks, run neural network in parallel
4. Ensemble predictions when confidence is low

This hybrid approach aims to get the best of both worlds:
- HypothesisEngine handles abstract reasoning via compositional search
- DSL handles simple geometric transformations perfectly
- Neural network learns complex pattern rules from data
"""

import torch
import torch.nn.functional as F
import json
import copy
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

# Import DSL components
from arc_solver import (
    ARCSolver, 
    HintGenerator, 
    ProgramSynthesizer,
    OPERATIONS,
    find_connected_components
)

# Import hypothesis engine
from core.hypothesis import HypothesisEngine

# Import neural components
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from config import get_config
from model import OctoTetrahedralModel


def get_tokenizer():
    if HAS_TIKTOKEN:
        return tiktoken.get_encoding("cl100k_base")
    else:
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 1000 for c in text]
            def decode(self, tokens):
                return ''.join(chr(t % 256) for t in tokens)
        return SimpleTokenizer()


def grid_to_text(grid: List[List[int]]) -> str:
    """Convert grid to text representation"""
    return '[' + '|'.join(','.join(str(c) for c in row) for row in grid) + ']'


def text_to_grid(text: str) -> Optional[List[List[int]]]:
    """Convert text back to grid"""
    try:
        text = text.strip().strip('[]')
        rows = text.split('|')
        grid = []
        for row in rows:
            cells = [int(c) for c in row.split(',')]
            grid.append(cells)
        return grid
    except:
        return None


def format_task_prompt(task: Dict, test_idx: int = 0, include_answer: bool = False) -> str:
    """Format ARC task as text prompt"""
    prompt = "Task: Transform input grid to output grid.\n\n"
    
    # Training examples
    for i, ex in enumerate(task['train']):
        prompt += f"Example {i+1}:\n"
        prompt += f"Input: {grid_to_text(ex['input'])}\n"
        prompt += f"Output: {grid_to_text(ex['output'])}\n\n"
    
    # Test input
    prompt += "Test:\n"
    prompt += f"Input: {grid_to_text(task['test'][test_idx]['input'])}\n"
    prompt += "Output: "
    
    if include_answer and 'output' in task['test'][test_idx]:
        prompt += grid_to_text(task['test'][test_idx]['output'])
    
    return prompt


class NeuralARCSolver:
    """Neural network based ARC solver using OctoTetrahedral model"""
    
    _DEFAULT_CHECKPOINT = str(Path(__file__).parent /
        'Archived_Projects/octotetrahedral_agi/checkpoints/arc/arc_step_2500.pt')

    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.config = get_config()
        self.device = device or self.config.device

        # Load model
        self.model = OctoTetrahedralModel(self.config)

        resolved = checkpoint_path or self._DEFAULT_CHECKPOINT
        if Path(resolved).exists():
            checkpoint = torch.load(resolved, map_location=self.device)
            result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            matched = len(self.model.state_dict()) - len(result.missing_keys)
            print(f"Loaded checkpoint from {resolved} ({matched}/{len(self.model.state_dict())} layers matched)")
        else:
            print(f"Warning: Checkpoint not found at {resolved}, using random initialization")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = get_tokenizer()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.3,
        time_budget: float = 3.0
    ) -> str:
        """Generate output from prompt with a hard time budget (seconds)"""
        import time
        deadline = time.time() + time_budget

        input_tokens = self.tokenizer.encode(prompt)
        # Truncate prompt if too long
        if len(input_tokens) > 400:
            input_tokens = input_tokens[-400:]
        input_ids = torch.tensor([input_tokens]).to(self.device)

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if time.time() > deadline:
                    break

                if generated.shape[1] > 512:
                    context = generated[:, -512:]
                else:
                    context = generated

                output = self.model(input_ids=context)
                logits = output['logits'][:, -1, :] / temperature

                # Top-k sampling
                top_k = 50
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                # Stop at ] which marks end of grid
                decoded = self.tokenizer.decode([next_token.item()])
                if ']' in decoded:
                    break

        # Return only generated part
        generated_text = self.tokenizer.decode(generated[0, len(input_tokens):].tolist())
        return generated_text.strip()

    def solve(self, task: Dict, test_idx: int = 0, time_budget: float = 2.0) -> Optional[List[List[int]]]:
        """Solve a single ARC task within time_budget seconds"""
        import time
        deadline = time.time() + time_budget
        prompt = format_task_prompt(task, test_idx, include_answer=False)

        # Generate multiple candidates — stop early if time runs out
        candidates = []
        for temp in [0.1, 0.3]:
            remaining = deadline - time.time()
            if remaining <= 0.3:
                break
            output = self.generate(prompt, temperature=temp, time_budget=min(remaining - 0.1, 1.5))
            grid = text_to_grid(output)
            if grid is not None:
                candidates.append(grid)
        
        if not candidates:
            return None
        
        # Return most common prediction
        grid_strs = [str(g) for g in candidates]
        most_common = Counter(grid_strs).most_common(1)[0][0]
        
        for i, g in enumerate(candidates):
            if str(g) == most_common:
                return g
        
        return candidates[0]


class HybridARCSolver:
    """Hybrid solver: HypothesisEngine → DSL → Neural ensemble"""

    def __init__(
        self,
        neural_checkpoint: str = None,
        use_neural_fallback: bool = True,
        device: str = None
    ):
        # Hypothesis engine (few-shot + composition search)
        self.hypothesis_engine = HypothesisEngine(
            max_composition_depth=3, timeout_seconds=15.0,
        )

        # DSL solver
        self.dsl_solver = ARCSolver()
        self.hint_gen = HintGenerator()

        # Neural solver (optional)
        self.use_neural = use_neural_fallback
        if use_neural_fallback:
            try:
                self.neural_solver = NeuralARCSolver(neural_checkpoint, device)
            except Exception as e:
                print(f"Warning: Could not load neural solver: {e}")
                self.use_neural = False
                self.neural_solver = None
        else:
            self.neural_solver = None

    def solve(self, task: Dict) -> List[List[List[int]]]:
        """Solve task using 3-stage pipeline, return up to 2 predictions"""
        train = task['train']
        test_input = task['test'][0]['input']

        predictions = []

        # Stage 1: HypothesisEngine (few-shot abstraction + composition)
        try:
            hyp_result = self.hypothesis_engine.solve(task)
            if hyp_result['solved'] and hyp_result['prediction'] is not None:
                predictions.append(hyp_result['prediction'])
        except Exception:
            pass

        # Stage 2: DSL solver (may find solutions hypothesis engine missed)
        dsl_preds = self.dsl_solver.solve(task)
        for pred in dsl_preds:
            if pred != test_input and pred not in predictions:
                predictions.append(pred)

        # Stage 3: Neural fallback — trigger when neither hypothesis nor DSL
        # produced a confident answer (not just when DSL gave up entirely)
        has_confident = len(predictions) > 0 and predictions[0] != test_input
        if not has_confident and self.use_neural and self.neural_solver:
            try:
                neural_pred = self.neural_solver.solve(task, time_budget=2.0)
                if neural_pred is not None and neural_pred not in predictions:
                    predictions.append(neural_pred)
            except Exception:
                pass

        # Deduplicate
        unique = []
        seen = set()
        for pred in predictions:
            key = str(pred)
            if key not in seen:
                seen.add(key)
                unique.append(pred)
        
        # Fallback: return input
        if not unique:
            unique.append(copy.deepcopy(test_input))
        
        return unique[:2]
    
    def evaluate(
        self, 
        data_dir: str, 
        max_tasks: int = 100, 
        split: str = 'training'
    ) -> Dict:
        """Evaluate solver on dataset"""
        task_dir = Path(data_dir) / split
        task_files = sorted(task_dir.glob('*.json'))[:max_tasks]
        
        results = {'total': 0, 'pass1': 0, 'pass2': 0, 'details': []}
        
        for task_file in task_files:
            with open(task_file) as f:
                task = json.load(f)
            
            if 'output' not in task['test'][0]:
                continue
            
            results['total'] += 1
            ground_truth = task['test'][0]['output']
            predictions = self.solve(task)
            
            detail = {
                'task_id': task_file.stem,
                'passed': False,
                'pass_type': None
            }
            
            if predictions:
                if predictions[0] == ground_truth:
                    results['pass1'] += 1
                    results['pass2'] += 1
                    detail['passed'] = True
                    detail['pass_type'] = 'pass@1'
                    print(f"  {task_file.stem}")
                elif len(predictions) > 1 and predictions[1] == ground_truth:
                    results['pass2'] += 1
                    detail['passed'] = True
                    detail['pass_type'] = 'pass@2'
                    print(f"  {task_file.stem} (pass@2)")
                else:
                    print(f"  {task_file.stem}")
            else:
                print(f"  {task_file.stem} (no pred)")
            
            results['details'].append(detail)
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(Path.home() / 'ARC_AMD_TRANSFER/data/ARC-AGI/data'))
    parser.add_argument('--max-tasks', type=int, default=100)
    parser.add_argument('--split', default='training')
    parser.add_argument('--neural-checkpoint', default=None)
    parser.add_argument('--no-neural', action='store_true', help='Disable neural fallback')
    args = parser.parse_args()
    
    print("=" * 70)
    print("ARC Hybrid Solver")
    print("DSL Program Synthesis + Neural Network Fallback")
    print("=" * 70)
    print()
    
    solver = HybridARCSolver(
        neural_checkpoint=args.neural_checkpoint,
        use_neural_fallback=not args.no_neural
    )
    
    results = solver.evaluate(args.data_dir, args.max_tasks, args.split)
    
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    if results['total'] > 0:
        p1 = results['pass1'] / results['total'] * 100
        p2 = results['pass2'] / results['total'] * 100
        print(f"Tasks:  {results['total']}")
        print(f"Pass@1: {results['pass1']}/{results['total']} ({p1:.1f}%)")
        print(f"Pass@2: {results['pass2']}/{results['total']} ({p2:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
