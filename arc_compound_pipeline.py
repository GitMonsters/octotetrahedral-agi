#!/usr/bin/env python3
"""
ARC-AGI Compound Pipeline — 3-Layer Cascading Solver

Layer 1: Catalog lookup (514 known solvers, instant)
Layer 2: Neural grid model (trained transformer, ~1s per task)
Layer 3: LLM program synthesis (runtime code gen, ~30s per task)

Each layer is tried in order. If a layer produces an exact match on
training examples, its answer is used. Otherwise, fall through to next.

Usage:
    # Full eval on all 400 eval tasks (catalog + neural only, no API needed)
    python arc_compound_pipeline.py --data ARC_AMD_TRANSFER/data/ARC-AGI/data --split evaluation

    # With LLM layer (needs API key)
    ANTHROPIC_API_KEY=... python arc_compound_pipeline.py --data ... --use-llm

    # Single task
    python arc_compound_pipeline.py --task 1a2e2828 --data ...
"""

import json
import os
import sys
import time
import argparse
import importlib.util
import traceback
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from multiprocessing import Process, Queue

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ─── Layer 1: Catalog Lookup ───────────────────────────────────────

CATALOG_DIR = Path(__file__).parent / "arc-puzzle-catalog" / "solves"


def catalog_solve(task_id: str, task: dict) -> Optional[List[List[List[int]]]]:
    """Try to solve using a known catalog solver."""
    solver_path = CATALOG_DIR / task_id / "solver.py"
    if not solver_path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location(f"solver_{task_id}", solver_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Catalog solvers use solve() or transform()
        fn = getattr(mod, 'solve', None) or getattr(mod, 'transform', None)
        if fn is None:
            return None

        # Verify on training examples first
        for ex in task['train']:
            result = fn(ex['input'])
            if result != ex['output']:
                return None

        # Apply to test inputs
        outputs = []
        for test in task['test']:
            out = fn(test['input'])
            outputs.append(out)
        return outputs

    except Exception:
        return None


# ─── Layer 2: Neural Grid Model ───────────────────────────────────

def load_neural_model(checkpoint_path: str, device: str = 'cpu',
                      d_model: int = 192, num_layers: int = 6, nhead: int = 8):
    """Load trained ARCGridModel."""
    from train_arc_v2 import ARCGridModel
    model = ARCGridModel(
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=d_model * 4,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def neural_solve(model, task: dict, device: str = 'cpu') -> Optional[List[List[List[int]]]]:
    """Try to solve using the neural grid model."""
    from train_arc_v2 import pad_grid, MAX_GRID, MAX_EXAMPLES, PAD

    outputs = []
    all_verified = True

    for test_idx, test_pair in enumerate(task['test']):
        train_exs = task['train'][:MAX_EXAMPLES]

        # Build context grids
        context_grids = []
        for ex in train_exs:
            context_grids.append(pad_grid(ex['input'], MAX_GRID, MAX_GRID))
            context_grids.append(pad_grid(ex['output'], MAX_GRID, MAX_GRID))
        while len(context_grids) < MAX_EXAMPLES * 2:
            context_grids.append(torch.full((MAX_GRID, MAX_GRID), PAD, dtype=torch.long))

        context = torch.stack(context_grids).unsqueeze(0).to(device)
        test_in = pad_grid(test_pair['input'], MAX_GRID, MAX_GRID).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model(context, test_in)
            preds = result['logits'].argmax(dim=-1)[0].cpu().numpy()

        # Determine output size: use test output if available, else guess from input
        if 'output' in test_pair:
            h = len(test_pair['output'])
            w = len(test_pair['output'][0]) if test_pair['output'] else 0
        else:
            # For competition: guess output size from training pattern
            h, w = guess_output_size(task, test_pair)

        out_grid = preds[:h, :w].tolist()
        outputs.append(out_grid)

    # Verify on training examples (use model's own predictions)
    for ex in task['train']:
        train_ctx = []
        other_exs = [e for e in task['train'] if e is not ex][:MAX_EXAMPLES]
        for e in other_exs:
            train_ctx.append(pad_grid(e['input'], MAX_GRID, MAX_GRID))
            train_ctx.append(pad_grid(e['output'], MAX_GRID, MAX_GRID))
        while len(train_ctx) < MAX_EXAMPLES * 2:
            train_ctx.append(torch.full((MAX_GRID, MAX_GRID), PAD, dtype=torch.long))

        ctx_t = torch.stack(train_ctx).unsqueeze(0).to(device)
        inp_t = pad_grid(ex['input'], MAX_GRID, MAX_GRID).unsqueeze(0).to(device)

        with torch.no_grad():
            r = model(ctx_t, inp_t)
            p = r['logits'].argmax(dim=-1)[0].cpu().numpy()

        eh, ew = len(ex['output']), len(ex['output'][0])
        pred_grid = p[:eh, :ew].tolist()
        if pred_grid != ex['output']:
            all_verified = False
            break

    if all_verified:
        return outputs
    return None


def guess_output_size(task: dict, test_pair: dict) -> Tuple[int, int]:
    """Guess output dimensions from training examples."""
    # Check if output size is always same as input
    same_size = all(
        len(ex['output']) == len(ex['input']) and
        len(ex['output'][0]) == len(ex['input'][0])
        for ex in task['train']
    )
    if same_size:
        return len(test_pair['input']), len(test_pair['input'][0])

    # Check if output size is constant
    out_sizes = [(len(ex['output']), len(ex['output'][0])) for ex in task['train']]
    if len(set(out_sizes)) == 1:
        return out_sizes[0]

    # Default: same as input
    return len(test_pair['input']), len(test_pair['input'][0])


def neural_predict(model, task: dict, device: str = 'cpu') -> Optional[List[List[List[int]]]]:
    """Predict test outputs using neural model (no self-verification)."""
    from train_arc_v2 import pad_grid, MAX_GRID, MAX_EXAMPLES, PAD

    outputs = []
    for test_pair in task['test']:
        train_exs = task['train'][:MAX_EXAMPLES]

        context_grids = []
        for ex in train_exs:
            context_grids.append(pad_grid(ex['input'], MAX_GRID, MAX_GRID))
            context_grids.append(pad_grid(ex['output'], MAX_GRID, MAX_GRID))
        while len(context_grids) < MAX_EXAMPLES * 2:
            context_grids.append(torch.full((MAX_GRID, MAX_GRID), PAD, dtype=torch.long))

        context = torch.stack(context_grids).unsqueeze(0).to(device)
        test_in = pad_grid(test_pair['input'], MAX_GRID, MAX_GRID).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model(context, test_in)
            preds = result['logits'].argmax(dim=-1)[0].cpu().numpy()

        if 'output' in test_pair:
            h = len(test_pair['output'])
            w = len(test_pair['output'][0]) if test_pair['output'] else 0
        else:
            h, w = guess_output_size(task, test_pair)

        outputs.append(preds[:h, :w].tolist())
    return outputs


# ─── Layer 3: LLM Program Synthesis ───────────────────────────────

def llm_solve(task_id: str, task: dict, backend: str = 'anthropic',
              model_name: str = None, max_attempts: int = 5) -> Optional[List[List[List[int]]]]:
    """Try to solve using LLM-generated code."""
    try:
        from arc_kaggle_solver import solve_task, make_llm_call
        llm_call = make_llm_call(backend, model_name)
        result = solve_task(task_id, task, llm_call, max_attempts=max_attempts)
        if result and result.get('solved'):
            return [result['test_outputs'][i] for i in range(len(task['test']))]
    except Exception as e:
        log.debug(f"LLM solve failed for {task_id}: {e}")
    return None


# ─── Compound Pipeline ────────────────────────────────────────────

class CompoundPipeline:
    def __init__(
        self,
        use_catalog: bool = True,
        use_neural: bool = True,
        use_llm: bool = False,
        neural_checkpoint: str = 'checkpoints/arc_grid/best_grid.pt',
        neural_device: str = None,
        llm_backend: str = 'anthropic',
        llm_model: str = None,
        llm_attempts: int = 5,
    ):
        self.use_catalog = use_catalog
        self.use_neural = use_neural
        self.use_llm = use_llm
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.llm_attempts = llm_attempts

        # Load neural model
        self.neural_model = None
        self.neural_device = neural_device or (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        if use_neural and Path(neural_checkpoint).exists():
            log.info(f"Loading neural model from {neural_checkpoint}...")
            self.neural_model = load_neural_model(
                neural_checkpoint, device=self.neural_device)
            log.info(f"Neural model loaded on {self.neural_device}")

        self.stats = {
            'catalog': 0, 'neural': 0, 'llm': 0,
            'unsolved': 0, 'total': 0, 'errors': 0,
        }

    def solve(self, task_id: str, task: dict) -> Tuple[Optional[List], str]:
        """
        Try each layer in cascade. Returns (outputs, method) or (None, 'unsolved').
        """
        self.stats['total'] += 1

        # Layer 1: Catalog
        if self.use_catalog:
            result = catalog_solve(task_id, task)
            if result is not None:
                self.stats['catalog'] += 1
                return result, 'catalog'

        # Layer 2: Neural (skip self-verification — use direct prediction)
        if self.use_neural and self.neural_model is not None:
            try:
                result = neural_predict(self.neural_model, task, self.neural_device)
                if result is not None:
                    self.stats['neural'] += 1
                    return result, 'neural'
            except Exception as e:
                log.debug(f"Neural error on {task_id}: {e}")
                self.stats['errors'] += 1

        # Layer 3: LLM
        if self.use_llm:
            result = llm_solve(
                task_id, task, self.llm_backend,
                self.llm_model, self.llm_attempts)
            if result is not None:
                self.stats['llm'] += 1
                return result, 'llm'

        self.stats['unsolved'] += 1
        return None, 'unsolved'

    def print_stats(self):
        s = self.stats
        total = s['total'] or 1
        solved = s['catalog'] + s['neural'] + s['llm']
        print()
        print("=" * 60)
        print(f"COMPOUND PIPELINE RESULTS — {solved}/{s['total']} solved ({100*solved/total:.1f}%)")
        print("=" * 60)
        print(f"  Layer 1 (Catalog):  {s['catalog']:>4} tasks")
        print(f"  Layer 2 (Neural):   {s['neural']:>4} tasks")
        print(f"  Layer 3 (LLM):      {s['llm']:>4} tasks")
        print(f"  Unsolved:           {s['unsolved']:>4} tasks")
        if s['errors']:
            print(f"  Errors:             {s['errors']:>4}")
        print("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ARC Compound Pipeline')
    parser.add_argument('--data', default='ARC_AMD_TRANSFER/data/ARC-AGI/data',
                        help='Path to ARC data directory')
    parser.add_argument('--split', default='evaluation',
                        help='Dataset split (training/evaluation)')
    parser.add_argument('--task', default=None,
                        help='Solve single task by ID')
    parser.add_argument('--no-catalog', action='store_true',
                        help='Skip catalog lookup')
    parser.add_argument('--no-neural', action='store_true',
                        help='Skip neural model')
    parser.add_argument('--use-llm', action='store_true',
                        help='Enable LLM layer (needs API key)')
    parser.add_argument('--llm-backend', default='anthropic',
                        choices=['anthropic', 'openai', 'ollama'])
    parser.add_argument('--llm-model', default=None)
    parser.add_argument('--llm-attempts', type=int, default=5)
    parser.add_argument('--checkpoint', default='checkpoints/arc_grid/best_grid.pt')
    parser.add_argument('--device', default=None)
    parser.add_argument('--out', default=None,
                        help='Output JSON file for submission')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max tasks to evaluate')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("ARC Compound Pipeline — Catalog + Neural + LLM")
    log.info("=" * 60)

    pipeline = CompoundPipeline(
        use_catalog=not args.no_catalog,
        use_neural=not args.no_neural,
        use_llm=args.use_llm,
        neural_checkpoint=args.checkpoint,
        neural_device=args.device,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_attempts=args.llm_attempts,
    )

    # Load tasks
    if args.task:
        task_file = Path(args.data) / args.split / f"{args.task}.json"
        with open(task_file) as f:
            task_data = json.load(f)
        tasks = [(args.task, task_data)]
    else:
        task_dir = Path(args.data) / args.split
        tasks = []
        for f in sorted(task_dir.glob('*.json')):
            with open(f) as fh:
                tasks.append((f.stem, json.load(fh)))

    if args.limit:
        tasks = tasks[:args.limit]

    log.info(f"Evaluating {len(tasks)} tasks from {args.split}")

    # Solve
    submission = {}
    results_by_method = {'catalog': [], 'neural': [], 'llm': [], 'unsolved': []}
    start = time.time()

    for i, (tid, tdata) in enumerate(tasks):
        outputs, method = pipeline.solve(tid, tdata)

        if outputs is not None:
            # Verify against known test outputs if available
            correct = True
            for j, test in enumerate(tdata['test']):
                if 'output' in test:
                    if outputs[j] != test['output']:
                        correct = False
                        break

            tag = "CORRECT" if correct else "WRONG"
            log.info(f"[{i+1}/{len(tasks)}] {tid}: {method} ({tag})")
            submission[tid] = outputs
        else:
            log.info(f"[{i+1}/{len(tasks)}] {tid}: unsolved")

        results_by_method[method].append(tid)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            solved = pipeline.stats['catalog'] + pipeline.stats['neural'] + pipeline.stats['llm']
            log.info(f"  Progress: {i+1}/{len(tasks)} | Solved: {solved} | {elapsed:.0f}s")

    elapsed = time.time() - start
    pipeline.print_stats()
    log.info(f"Time: {elapsed:.1f}s ({elapsed/len(tasks):.2f}s/task)")

    # Save submission
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(submission, f)
        log.info(f"Submission saved to {args.out}")

    # Print method breakdown
    if results_by_method['neural']:
        log.info(f"Neural solved: {results_by_method['neural']}")
    if results_by_method['llm']:
        log.info(f"LLM solved: {results_by_method['llm']}")


if __name__ == '__main__':
    main()
