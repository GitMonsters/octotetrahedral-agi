#!/usr/bin/env python3
"""
ARC Solver using local MLX models (TinyLlama, Qwen)
Works on Mac with Apple Silicon
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import time

try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available. Install with: pip install mlx-lm")


def grid_to_str(grid: List[List[int]]) -> str:
    """Convert grid to string representation"""
    return str(grid)


def str_to_grid(s: str) -> Optional[List[List[int]]]:
    """Parse string back to grid"""
    try:
        # Clean up the string
        s = s.strip()
        # Find the first complete grid pattern [[...]]
        start = s.find('[[')
        if start == -1:
            return None
        
        # Find matching end
        depth = 0
        end = start
        for i, c in enumerate(s[start:]):
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break
        
        grid_str = s[start:end]
        grid = eval(grid_str)
        
        # Validate it's a 2D list of integers
        if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
            return [[int(c) for c in row] for row in grid]
        return None
    except:
        return None


def format_task_prompt(task: Dict, test_idx: int = 0) -> str:
    """Format ARC task as prompt"""
    lines = ["Learn the pattern from examples and predict the output grid.\n"]
    
    # Add training examples
    for i, ex in enumerate(task['train']):
        lines.append(f"Example {i+1}:")
        lines.append(f"Input: {grid_to_str(ex['input'])}")
        lines.append(f"Output: {grid_to_str(ex['output'])}")
        lines.append("")
    
    # Add test input
    lines.append("Test:")
    lines.append(f"Input: {grid_to_str(task['test'][test_idx]['input'])}")
    lines.append("Output:")
    
    return '\n'.join(lines)


class MLXARCSolver:
    """ARC solver using MLX local models"""
    
    def __init__(self, model_name: str = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"):
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        
        print(f"Loading model: {model_name}")
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        print("Model loaded!")
    
    def solve(self, task: Dict, test_idx: int = 0, max_tokens: int = 100, temperature: float = 0.3) -> Optional[List[List[int]]]:
        """Solve a single ARC task"""
        prompt = format_task_prompt(task, test_idx)
        
        # Create sampler with temperature
        sampler = make_sampler(temp=temperature)
        
        # Generate response
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens,
            sampler=sampler
        )
        
        # Parse grid from response
        grid = str_to_grid(response)
        return grid
    
    def solve_with_retries(self, task: Dict, test_idx: int = 0, num_retries: int = 3) -> List[List[List[int]]]:
        """Try multiple times with different temperatures"""
        predictions = []
        temps = [0.1, 0.3, 0.5]
        
        for i, temp in enumerate(temps[:num_retries]):
            pred = self.solve(task, test_idx, temperature=temp)
            if pred is not None and pred not in predictions:
                predictions.append(pred)
        
        return predictions


def evaluate(
    solver: MLXARCSolver,
    data_dir: str,
    split: str = 'training',
    max_tasks: int = 50
) -> Dict:
    """Evaluate solver on ARC tasks"""
    task_dir = Path(data_dir) / split
    task_files = sorted(task_dir.glob('*.json'))[:max_tasks]
    
    results = {'total': 0, 'pass1': 0, 'pass2': 0, 'failed': []}
    
    for task_file in task_files:
        with open(task_file) as f:
            task = json.load(f)
        
        if 'output' not in task['test'][0]:
            continue
        
        results['total'] += 1
        ground_truth = task['test'][0]['output']
        
        # Get predictions
        predictions = solver.solve_with_retries(task, test_idx=0, num_retries=2)
        
        if predictions:
            if predictions[0] == ground_truth:
                results['pass1'] += 1
                results['pass2'] += 1
                print(f"✓ {task_file.stem}")
            elif len(predictions) > 1 and predictions[1] == ground_truth:
                results['pass2'] += 1
                print(f"○ {task_file.stem} (pass@2)")
            else:
                print(f"✗ {task_file.stem}")
                results['failed'].append(task_file.stem)
        else:
            print(f"✗ {task_file.stem} (no valid output)")
            results['failed'].append(task_file.stem)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ARC Solver with local MLX models')
    parser.add_argument('--model', default='mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit',
                       help='MLX model to use')
    parser.add_argument('--data-dir', default=str(Path.home() / 'ARC_AMD_TRANSFER/data/ARC-AGI/data'))
    parser.add_argument('--max-tasks', type=int, default=50)
    parser.add_argument('--split', default='training')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ARC Solver - Local MLX Models")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_dir}")
    print()
    
    solver = MLXARCSolver(args.model)
    
    print("\nRunning evaluation...")
    start = time.time()
    results = evaluate(solver, args.data_dir, args.split, args.max_tasks)
    elapsed = time.time() - start
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    if results['total'] > 0:
        p1 = results['pass1'] / results['total'] * 100
        p2 = results['pass2'] / results['total'] * 100
        print(f"Tasks:  {results['total']}")
        print(f"Pass@1: {results['pass1']}/{results['total']} ({p1:.1f}%)")
        print(f"Pass@2: {results['pass2']}/{results['total']} ({p2:.1f}%)")
        print(f"Time:   {elapsed:.1f}s ({elapsed/results['total']:.1f}s per task)")
    print("=" * 60)


if __name__ == "__main__":
    main()
