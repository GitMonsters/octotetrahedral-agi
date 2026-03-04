#!/usr/bin/env python3
"""
Combined ARC-AGI evaluation pipeline.

Runs multiple solvers in cascade order and picks the best prediction:
1. arc_solver.py (fast symbolic DSL)
2. arc_synth.py (program synthesis)
3. arc_trm.py (neural TRM with test-time training)

Usage:
  python arc_combined_eval.py --eval-dir ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation
"""

import json
import glob
import os
import sys
import time
import copy
import argparse
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))


def load_task(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def grid_eq(a, b) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(
        len(ra) == len(rb) and all(ca == cb for ca, cb in zip(ra, rb))
        for ra, rb in zip(a, b)
    )


def is_identity(pred, test_input) -> bool:
    return grid_eq(pred, test_input)


def cell_accuracy(pred, expected) -> float:
    if pred is None or expected is None:
        return 0.0
    if len(pred) != len(expected):
        return 0.0
    total = 0
    correct = 0
    for i in range(len(expected)):
        if len(pred[i]) != len(expected[i]):
            return 0.0
        for j in range(len(expected[i])):
            total += 1
            if pred[i][j] == expected[i][j]:
                correct += 1
    return correct / total if total > 0 else 0.0


class CombinedSolver:
    def __init__(self, use_arc_solver=True, use_arc_synth=True, use_trm=False, trm_checkpoint=None):
        self.solvers = []
        self.arc_solver = None
        self.arc_synth = None
        self.trm_model = None

        if use_arc_solver:
            try:
                from arc_solver import ARCSolver
                self.arc_solver = ARCSolver()
                self.solvers.append('arc_solver')
                print("  ✓ arc_solver loaded")
            except Exception as e:
                print(f"  ✗ arc_solver: {e}")

        if use_arc_synth:
            try:
                from arc_synth import ARCSynthesizer
                self.arc_synth = ARCSynthesizer(max_time=15.0)
                self.solvers.append('arc_synth')
                print("  ✓ arc_synth loaded")
            except Exception as e:
                print(f"  ✗ arc_synth: {e}")

        if use_trm and trm_checkpoint and os.path.exists(trm_checkpoint):
            try:
                import torch
                from arc_trm import TinyRecursiveModel, solve_task_trm, get_device
                self.trm_device = get_device()
                ckpt = torch.load(trm_checkpoint, map_location='cpu', weights_only=False)
                config = ckpt['config']
                self.trm_model = TinyRecursiveModel(
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    h_cycles=config['h_cycles'],
                    l_cycles=config['l_cycles'],
                    max_puzzles=config.get('num_puzzles', 1100) + 100,
                    use_attention=config.get('use_attention', True),
                    num_heads=config.get('num_heads', 4),
                ).to(self.trm_device)
                state_key = 'ema_state' if 'ema_state' in ckpt else 'model_state'
                self.trm_model.load_state_dict(ckpt[state_key], strict=False)
                self.trm_original_state = copy.deepcopy(self.trm_model.state_dict())
                self.solve_task_trm = solve_task_trm
                self.solvers.append('trm')
                params = sum(p.numel() for p in self.trm_model.parameters())
                print(f"  ✓ TRM loaded ({params:,} params)")
            except Exception as e:
                print(f"  ✗ TRM: {e}")

        print(f"  Active solvers: {self.solvers}")

    def solve(self, task: dict, test_idx: int = 0) -> List[dict]:
        test_input = task['test'][test_idx]['input']
        predictions = []

        if self.arc_solver:
            try:
                preds = self.arc_solver.solve(task, max_time=10.0, test_idx=test_idx)
                for i, pred in enumerate(preds):
                    if not is_identity(pred, test_input):
                        predictions.append({
                            'grid': pred, 'solver': 'arc_solver',
                            'confidence': 0.9 - 0.1 * i,
                        })
            except:
                pass

        if self.arc_synth:
            try:
                all_preds = self.arc_synth.synthesize(task)
                pred = all_preds[test_idx] if test_idx < len(all_preds) else None
                if pred and not is_identity(pred, test_input):
                    predictions.append({
                        'grid': pred, 'solver': 'arc_synth',
                        'confidence': 0.85,
                    })
            except:
                pass

        if self.trm_model:
            try:
                import torch
                self.trm_model.load_state_dict(self.trm_original_state, strict=False)
                trm_preds = self.solve_task_trm(
                    self.trm_model, task, self.trm_device,
                    test_idx=test_idx, ttt_steps=150, ttt_lr=3e-4,
                )
                for i, pred in enumerate(trm_preds):
                    if not is_identity(pred, test_input):
                        predictions.append({
                            'grid': pred, 'solver': 'trm',
                            'confidence': 0.7 - 0.1 * i,
                        })
            except:
                pass

        # Deduplicate and rank
        seen = set()
        unique = []
        for p in sorted(predictions, key=lambda x: x['confidence'], reverse=True):
            key = str(p['grid'])
            if key not in seen:
                seen.add(key)
                unique.append(p)

        if not unique:
            unique.append({
                'grid': copy.deepcopy(test_input),
                'solver': 'identity', 'confidence': 0.0,
            })

        return unique[:2]


def evaluate(
    eval_dir: str,
    use_arc_solver: bool = True,
    use_arc_synth: bool = True,
    use_trm: bool = False,
    trm_checkpoint: str = 'arc_trm_pretrained.pt',
    max_tasks: int = 0,
    verbose: bool = False,
    output_file: str = '',
):
    print("=" * 60)
    print("Combined ARC-AGI Evaluation")
    print("=" * 60)
    print("\nLoading solvers...")

    solver = CombinedSolver(
        use_arc_solver=use_arc_solver,
        use_arc_synth=use_arc_synth,
        use_trm=use_trm,
        trm_checkpoint=trm_checkpoint,
    )

    task_files = sorted(glob.glob(os.path.join(eval_dir, '*.json')))
    if max_tasks > 0:
        task_files = task_files[:max_tasks]

    print(f"\nEvaluating {len(task_files)} tasks...")

    results = {}
    solver_counts = {}
    near_misses = []

    for i, tf in enumerate(task_files):
        task_id = os.path.basename(tf).replace('.json', '')
        task = load_task(tf)

        task_correct_p1 = True
        task_correct_p2 = True

        for test_idx in range(len(task['test'])):
            preds = solver.solve(task, test_idx=test_idx)
            test_input = task['test'][test_idx]['input']

            if 'output' in task['test'][test_idx]:
                expected = task['test'][test_idx]['output']
                p1_ok = grid_eq(preds[0]['grid'], expected)
                p2_ok = p1_ok or (len(preds) > 1 and grid_eq(preds[1]['grid'], expected))

                if not p1_ok:
                    task_correct_p1 = False
                if not p2_ok:
                    task_correct_p2 = False

                best_acc = max(cell_accuracy(p['grid'], expected) for p in preds)
                if best_acc > 0.85 and not p2_ok:
                    near_misses.append((task_id, test_idx, best_acc, preds[0]['solver']))

                if p1_ok or p2_ok:
                    ws = preds[0]['solver'] if p1_ok else preds[1]['solver']
                    solver_counts[ws] = solver_counts.get(ws, 0) + 1
                    print(f"  {'✓' if p1_ok else '✓@2'} {task_id} (by {ws})")
                elif verbose:
                    print(f"  ✗ {task_id} (best: {preds[0]['solver']}, acc: {best_acc:.1%})")

        results[task_id] = {'pass1': task_correct_p1, 'pass2': task_correct_p2}

        if (i + 1) % 20 == 0:
            n = i + 1
            p1 = sum(1 for v in results.values() if v['pass1'])
            p2 = sum(1 for v in results.values() if v['pass2'])
            print(f"  [{n}/{len(task_files)}] P@1: {p1}/{n} P@2: {p2}/{n}")

    n = len(task_files)
    p1 = sum(1 for v in results.values() if v['pass1'])
    p2 = sum(1 for v in results.values() if v['pass2'])

    print(f"\n{'='*60}")
    print(f"Results: Pass@1={p1}/{n} ({100*p1/n:.1f}%)  Pass@2={p2}/{n} ({100*p2/n:.1f}%)")
    if solver_counts:
        print(f"Solvers: {solver_counts}")
    if near_misses:
        near_misses.sort(key=lambda x: -x[2])
        print(f"\nNear misses (>85% cell acc, not solved):")
        for tid, tidx, acc, sn in near_misses[:10]:
            print(f"  {tid}[{tidx}]: {acc:.1%} ({sn})")
    print("=" * 60)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump({k: v['pass1'] for k, v in results.items()}, f, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', default='ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation')
    parser.add_argument('--max-tasks', type=int, default=0)
    parser.add_argument('--no-arc-solver', action='store_true')
    parser.add_argument('--no-arc-synth', action='store_true')
    parser.add_argument('--use-trm', action='store_true')
    parser.add_argument('--trm-checkpoint', default='arc_trm_pretrained.pt')
    parser.add_argument('--ttt-steps', type=int, default=200)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    evaluate(
        eval_dir=args.eval_dir,
        use_arc_solver=not args.no_arc_solver,
        use_arc_synth=not args.no_arc_synth,
        use_trm=args.use_trm,
        trm_checkpoint=args.trm_checkpoint,
        max_tasks=args.max_tasks,
        verbose=args.verbose,
        output_file=args.output,
    )
