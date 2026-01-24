#!/usr/bin/env python3
"""
Integrated ARC Solver - OctoTetrahedral AGI + Full Program Synthesis
====================================================================

Combines:
1. Full DSL from arc_program_synthesis.py (20+ operations)
2. Test-Time Training with Leave-One-Out
3. OctoTetrahedral AGI neural generation
4. Hierarchical Voting

Target: 50%+ accuracy by combining all approaches
"""

import sys
import json
import copy
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
from itertools import product, permutations

# Add paths for imports
sys.path.insert(0, str(Path.home() / "ARC_AMD_TRANSFER" / "code"))
sys.path.insert(0, str(Path.home() / "octotetrahedral_agi"))

import torch
import torch.nn.functional as F

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# Import full DSL from arc_program_synthesis
try:
    from arc_program_synthesis import (
        ARCProgram, OPERATIONS, ProgramSynthesizer, ARCSolverWithSynthesis
    )
    HAS_FULL_DSL = True
    print("Loaded full DSL from arc_program_synthesis.py")
except ImportError:
    HAS_FULL_DSL = False
    print("Warning: Could not load arc_program_synthesis.py, using fallback DSL")


# ============================================================================
# Fallback DSL (if arc_program_synthesis.py not available)
# ============================================================================

if not HAS_FULL_DSL:
    def identity(grid, **kwargs):
        return grid

    def rotate_90(grid, **kwargs):
        return [list(row) for row in zip(*grid[::-1])]

    def rotate_180(grid, **kwargs):
        return [row[::-1] for row in grid[::-1]]

    def rotate_270(grid, **kwargs):
        return [list(row) for row in zip(*grid)][::-1]

    def flip_h(grid, **kwargs):
        return [row[::-1] for row in grid]

    def flip_v(grid, **kwargs):
        return grid[::-1]

    def transpose(grid, **kwargs):
        return [list(row) for row in zip(*grid)]

    OPERATIONS = {
        'identity': identity,
        'rotate_90': rotate_90,
        'rotate_180': rotate_180,
        'rotate_270': rotate_270,
        'flip_h': flip_h,
        'flip_v': flip_v,
        'transpose': transpose,
    }

    class ARCProgram:
        def __init__(self, operations: List[Tuple[str, Dict]]):
            self.operations = operations

        def execute(self, grid):
            result = copy.deepcopy(grid)
            for op_name, params in self.operations:
                if op_name in OPERATIONS:
                    result = OPERATIONS[op_name](result, **params)
            return result

        def __repr__(self):
            return " -> ".join([f"{op}({p})" for op, p in self.operations])

        def __hash__(self):
            return hash(str(self.operations))


INVERSE_TRANSFORMS = {
    'identity': 'identity',
    'rotate_90': 'rotate_270',
    'rotate_180': 'rotate_180',
    'rotate_270': 'rotate_90',
    'flip_h': 'flip_h',
    'flip_v': 'flip_v',
    'transpose': 'transpose',
}


# ============================================================================
# TTT Solver with Full Program Synthesis
# ============================================================================

class TTTSolver:
    """
    Test-Time Training solver using full DSL.
    Uses ARCSolverWithSynthesis if available, otherwise falls back to basic synthesis.
    """

    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.transforms = ['identity', 'rotate_90', 'rotate_180', 'rotate_270',
                          'flip_h', 'flip_v', 'transpose']
        
        # Use full solver if available
        if HAS_FULL_DSL:
            self.base_solver = ARCSolverWithSynthesis(max_depth=max_depth, num_attempts=3)
        else:
            self.base_solver = None

    def solve(self, task: Dict, num_predictions=2) -> List[List[List[int]]]:
        """Solve with TTT approach using full DSL"""
        train = task['train']
        test_input = task['test'][0]['input']

        if HAS_FULL_DSL and self.base_solver:
            return self._solve_with_full_dsl(task, num_predictions)
        else:
            return self._solve_with_basic_dsl(task, num_predictions)

    def _solve_with_full_dsl(self, task: Dict, num_predictions: int) -> List:
        """Use ARCSolverWithSynthesis with geometric augmentation"""
        train = task['train']
        test_input = task['test'][0]['input']
        
        all_predictions = defaultdict(list)

        for trans_name in self.transforms:
            # Apply transformation to training and test data
            if trans_name == 'identity':
                trans_fn = lambda x: x
                inv_fn = lambda x: x
            else:
                trans_fn = OPERATIONS[trans_name]
                inv_name = INVERSE_TRANSFORMS[trans_name]
                inv_fn = OPERATIONS[inv_name]

            # Transform entire task
            trans_task = {
                'train': [
                    {'input': trans_fn(ex['input']), 'output': trans_fn(ex['output'])}
                    for ex in train
                ],
                'test': [{'input': trans_fn(test_input)}]
            }

            # Solve transformed task
            try:
                preds = self.base_solver.solve(trans_task)
                for pred in preds:
                    # Apply inverse transformation
                    final_pred = inv_fn(pred)
                    all_predictions[trans_name].append(final_pred)
            except Exception as e:
                pass

        # Hierarchical voting
        return self._hierarchical_vote(all_predictions, num_predictions)

    def _solve_with_basic_dsl(self, task: Dict, num_predictions: int) -> List:
        """Fallback solver with basic geometric operations"""
        train = task['train']
        test_input = task['test'][0]['input']

        all_predictions = defaultdict(list)

        for trans_name in self.transforms:
            if trans_name == 'identity':
                trans_fn = lambda x: x
                inv_fn = lambda x: x
            else:
                trans_fn = OPERATIONS[trans_name]
                inv_name = INVERSE_TRANSFORMS[trans_name]
                inv_fn = OPERATIONS[inv_name]

            # Transform training examples
            trans_train = []
            for ex in train:
                trans_train.append({
                    'input': trans_fn(ex['input']),
                    'output': trans_fn(ex['output'])
                })

            # Transform test input
            trans_test = trans_fn(test_input)

            # Try each single operation
            for op_name in OPERATIONS.keys():
                try:
                    # Check if operation works on training examples
                    matches_all = True
                    for ex in trans_train:
                        pred = OPERATIONS[op_name](ex['input'])
                        if pred != ex['output']:
                            matches_all = False
                            break
                    
                    if matches_all:
                        pred = OPERATIONS[op_name](trans_test)
                        final_pred = inv_fn(pred)
                        all_predictions[trans_name].append(final_pred)
                except:
                    pass

        return self._hierarchical_vote(all_predictions, num_predictions)

    def _hierarchical_vote(self, predictions_by_trans: Dict, top_k: int) -> List:
        """Two-stage voting"""
        # Stage 1: Top-3 per transformation
        top_per_trans = []
        for trans, preds in predictions_by_trans.items():
            if preds:
                # Take first 3 unique predictions
                seen = set()
                for pred in preds:
                    pred_str = str(pred)
                    if pred_str not in seen:
                        seen.add(pred_str)
                        top_per_trans.append(pred)
                    if len(seen) >= 3:
                        break

        if not top_per_trans:
            return []

        # Stage 2: Global voting
        counts = Counter(str(p) for p in top_per_trans)
        pred_map = {str(p): p for p in top_per_trans}
        top_strs = [s for s, _ in counts.most_common(top_k)]
        return [pred_map[s] for s in top_strs]


# ============================================================================
# OctoTetrahedral Neural Solver
# ============================================================================

class OctoTetrahedralSolver:
    """Neural solver using OctoTetrahedral AGI"""

    def __init__(self, checkpoint_path: str = None):
        self.model = None
        self.tokenizer = None
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load OctoTetrahedral model"""
        try:
            from config import get_config
            from model import OctoTetrahedralModel

            config = get_config()
            self.model = OctoTetrahedralModel(config)

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            if HAS_TIKTOKEN:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            print(f"Loaded OctoTetrahedral model from {checkpoint_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            self.model = None

    def solve(self, task: Dict, num_predictions=2) -> List[List[List[int]]]:
        """Solve using neural generation"""
        if self.model is None or self.tokenizer is None:
            return []

        train = task['train']
        test_input = task['test'][0]['input']

        # Format as prompt
        prompt = self._format_prompt(train, test_input)

        predictions = []
        for temp in [0.3, 0.5]:
            try:
                output = self._generate(prompt, temperature=temp)
                grid = self._parse_grid(output)
                if grid:
                    predictions.append(grid)
            except:
                pass

        return predictions[:num_predictions]

    def _format_prompt(self, train: List[Dict], test_input: List[List[int]]) -> str:
        """Format ARC task as text prompt"""
        parts = []
        for i, ex in enumerate(train):
            inp_str = self._grid_to_str(ex['input'])
            out_str = self._grid_to_str(ex['output'])
            parts.append(f"[{inp_str}]->[{out_str}]")

        test_str = self._grid_to_str(test_input)
        parts.append(f"[{test_str}]->")

        return " ".join(parts)

    def _grid_to_str(self, grid: List[List[int]]) -> str:
        """Convert grid to compact string"""
        return " | ".join(" ".join(str(c) for c in row) for row in grid)

    def _generate(self, prompt: str, temperature=0.3, max_tokens=100) -> str:
        """Generate output from model"""
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens]).to(self.device)

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_tokens):
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

                decoded = self.tokenizer.decode([next_token.item()])
                if ']' in decoded:
                    break

        return self.tokenizer.decode(generated[0, len(tokens):].tolist())

    def _parse_grid(self, text: str) -> Optional[List[List[int]]]:
        """Parse grid from generated text"""
        try:
            text = text.strip().rstrip(']')
            rows = text.split(' | ')
            return [[int(c) for c in row.split()] for row in rows]
        except:
            return None


# ============================================================================
# Integrated Solver
# ============================================================================

class IntegratedARCSolver:
    """
    Combines TTT/DSL solver with OctoTetrahedral AGI.

    Strategy:
    1. Run TTT solver (highest accuracy on geometric tasks)
    2. Run neural solver as secondary
    3. Combine predictions with weighted voting
    """

    def __init__(self, neural_checkpoint: str = None, max_depth: int = 2):
        self.ttt_solver = TTTSolver(max_depth=max_depth)
        self.neural_solver = OctoTetrahedralSolver(neural_checkpoint)
        self.use_neural = self.neural_solver.model is not None

    def solve(self, task: Dict) -> List[List[List[int]]]:
        """Solve task with integrated approach"""
        all_predictions = []

        # 1. TTT/DSL solver (primary)
        ttt_preds = self.ttt_solver.solve(task, num_predictions=2)
        for pred in ttt_preds:
            all_predictions.append(('ttt', pred, 1.0))  # Higher weight

        # 2. Neural solver (secondary)
        if self.use_neural:
            neural_preds = self.neural_solver.solve(task, num_predictions=2)
            for pred in neural_preds:
                all_predictions.append(('neural', pred, 0.5))  # Lower weight

        # 3. Fallback: return test input
        test_input = task['test'][0]['input']
        all_predictions.append(('fallback', copy.deepcopy(test_input), 0.1))

        # Vote on predictions
        return self._weighted_vote(all_predictions, top_k=2)

    def _weighted_vote(self, predictions: List[Tuple[str, List, float]], top_k: int) -> List:
        """Weighted voting on predictions"""
        # Count weighted votes
        votes = defaultdict(float)
        pred_map = {}

        for source, pred, weight in predictions:
            key = str(pred)
            votes[key] += weight
            pred_map[key] = pred

        # Sort by votes
        sorted_keys = sorted(votes.keys(), key=lambda k: votes[k], reverse=True)
        return [pred_map[k] for k in sorted_keys[:top_k]]


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_solver(solver, data_dir: str, max_tasks: int = 50, split: str = 'training'):
    """Evaluate solver on ARC tasks"""
    task_dir = Path(data_dir) / split
    task_files = sorted(task_dir.glob('*.json'))[:max_tasks]

    results = {'total': 0, 'correct': 0, 'pass_at_2': 0}

    for task_file in task_files:
        with open(task_file) as f:
            task = json.load(f)

        # Check if ground truth available
        if 'output' not in task['test'][0]:
            continue

        results['total'] += 1
        ground_truth = task['test'][0]['output']

        predictions = solver.solve(task)

        if predictions:
            # Check pass@1
            if predictions[0] == ground_truth:
                results['correct'] += 1
                results['pass_at_2'] += 1
                print(f"✓ {task_file.stem}")
            # Check pass@2
            elif len(predictions) > 1 and predictions[1] == ground_truth:
                results['pass_at_2'] += 1
                print(f"○ {task_file.stem} (pass@2)")
            else:
                print(f"✗ {task_file.stem}")
        else:
            print(f"✗ {task_file.stem} (no prediction)")

    return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Integrated ARC Solver')
    parser.add_argument('--data-dir', type=str,
                       default=str(Path.home() / 'ARC_AMD_TRANSFER' / 'data' / 'ARC-AGI' / 'data'),
                       help='ARC data directory')
    parser.add_argument('--checkpoint', type=str,
                       default=str(Path.home() / 'octotetrahedral_agi' / 'checkpoints' / 'arc' / 'arc_step_500.pt'),
                       help='OctoTetrahedral checkpoint')
    parser.add_argument('--max-tasks', type=int, default=50, help='Max tasks to evaluate')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'evaluation'])
    parser.add_argument('--ttt-only', action='store_true', help='Only use TTT solver')
    parser.add_argument('--max-depth', type=int, default=2, help='DSL max depth')

    args = parser.parse_args()

    print("="*70)
    print("Integrated ARC Solver")
    print("OctoTetrahedral AGI + Full Program Synthesis")
    print("="*70)
    print(f"DSL: {'Full (20+ ops)' if HAS_FULL_DSL else 'Basic (7 ops)'}")
    print()

    # Create solver
    if args.ttt_only:
        solver = TTTSolver(max_depth=args.max_depth)
        print("Using TTT solver only")
    else:
        solver = IntegratedARCSolver(args.checkpoint, max_depth=args.max_depth)
        print(f"Using integrated solver")
        print(f"  Neural model: {'loaded' if solver.use_neural else 'not loaded'}")

    print()

    # Evaluate
    results = evaluate_solver(
        solver,
        args.data_dir,
        max_tasks=args.max_tasks,
        split=args.split
    )

    # Summary
    print()
    print("="*70)
    print("Results")
    print("="*70)
    if results['total'] > 0:
        acc = results['correct'] / results['total'] * 100
        pass2 = results['pass_at_2'] / results['total'] * 100
        print(f"Tasks:    {results['total']}")
        print(f"Pass@1:   {results['correct']}/{results['total']} ({acc:.1f}%)")
        print(f"Pass@2:   {results['pass_at_2']}/{results['total']} ({pass2:.1f}%)")
    else:
        print("No tasks evaluated")
    print("="*70)


if __name__ == "__main__":
    main()
