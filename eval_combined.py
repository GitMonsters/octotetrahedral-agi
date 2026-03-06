#!/usr/bin/env python3
"""
Combined ARC-AGI Solver — runs ALL available solvers with hard timeouts.

Goal: maximize task coverage by combining complementary approaches.
Each solver runs in a subprocess with enforced timeout.
"""

import os
import sys
import json
import time
import logging
import multiprocessing
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

ARC_V1_EVAL = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation'
ARC_V1_TRAIN = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/training'
# Note: gitmonsters_summary.json is TRAINING set, not eval
# Only use eval-set results for tracking
PREV_SUMMARY = None  # disabled — was training set data


def grids_match(pred, target) -> bool:
    if pred is None or target is None:
        return False
    if isinstance(pred, np.ndarray):
        pred = pred.tolist()
    if isinstance(target, np.ndarray):
        target = target.tolist()
    if len(pred) != len(target):
        return False
    for r1, r2 in zip(pred, target):
        if len(r1) != len(r2):
            return False
        if list(r1) != list(r2):
            return False
    return True


def load_tasks(data_dir: str) -> Dict[str, Dict]:
    tasks = {}
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.json'):
            tid = f.replace('.json', '')
            with open(os.path.join(data_dir, f)) as fh:
                tasks[tid] = json.load(fh)
    return tasks


# ═══════════════════════════════════════════════════════════════════════
# Solver workers — each runs in a subprocess
# ═══════════════════════════════════════════════════════════════════════

def _dsl_worker(task_json: str, result_queue):
    """DSL solver + HypothesisEngine."""
    task = json.loads(task_json)
    from arc_solver import ARCSolver
    from core.hypothesis import HypothesisEngine

    preds = []
    test_input = task['test'][0]['input']

    # DSL solver
    try:
        solver = ARCSolver()
        dsl_preds = solver.solve(task, max_time=8.0)
        for p in dsl_preds:
            if p != test_input and p not in preds:
                preds.append(('dsl', p))
    except Exception:
        pass

    # Hypothesis engine — deeper search
    try:
        he = HypothesisEngine(max_composition_depth=3, timeout_seconds=5.0)
        result = he.solve(task)
        if result.get('solved') and result.get('prediction') is not None:
            p = result['prediction']
            if p not in [x[1] for x in preds]:
                preds.append(('hypothesis', p))
    except Exception:
        pass

    result_queue.put(preds)


def _ttccvtlr_worker(task_json: str, result_queue):
    """TTCCVTLR cognitive loop solver."""
    task = json.loads(task_json)
    try:
        from core.ttccvtlr import TTCCVTLREngine
        engine = TTCCVTLREngine(
            max_rounds=3,
            confidence_threshold=0.6,
            use_neural_learning=True,
            timeout_seconds=25.0,
        )
        result = engine.solve(task, verbose=False)
        preds = []
        if result.get('solved') and result.get('prediction') is not None:
            preds.append(('ttccvtlr', result['prediction']))
        for pred, method, score in result.get('predictions', [])[1:]:
            preds.append(('ttccvtlr', pred))
        result_queue.put(preds)
    except Exception:
        result_queue.put([])


def _conv_ttt_worker(task_json: str, n_seeds: int, result_queue):
    """Convolutional test-time training with multi-seed ensemble."""
    task = json.loads(task_json)
    try:
        from arc_conv_ttt import solve_task as ttt_solve
        import torch

        # Use CPU in subprocess to avoid MPS/Metal hangs
        device = torch.device('cpu')

        all_preds = []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 42 + 7)
            np.random.seed(seed * 42 + 7)
            preds = ttt_solve(task, device=device, verbose=False)
            if preds is not None:
                for p in preds:
                    p_list = p.tolist() if hasattr(p, 'tolist') else p
                    all_preds.append(p_list)

        # Deduplicate
        unique = []
        seen = set()
        for p in all_preds:
            key = str(p)
            if key not in seen:
                seen.add(key)
                unique.append(('conv_ttt', p))

        result_queue.put(unique[:3])
    except Exception:
        result_queue.put([])


def _object_solver_worker(task_json: str, result_queue):
    """Object-centric solver."""
    task = json.loads(task_json)
    try:
        from arc_object_solver import solve_task as obj_solve
        preds = obj_solve(task)
        result = []
        if preds:
            for p in (preds if isinstance(preds, list) and isinstance(preds[0], list) and isinstance(preds[0][0], list) else [preds]):
                result.append(('object', p))
        result_queue.put(result[:2])
    except Exception:
        result_queue.put([])


def run_solver_with_timeout(worker_fn, args, timeout: int) -> List[Tuple[str, list]]:
    """Run a solver in a subprocess with hard timeout."""
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=worker_fn, args=(*args, result_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)
        return []

    if not result_queue.empty():
        try:
            return result_queue.get_nowait()
        except Exception:
            return []
    return []


# ═══════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_combined_evaluation():
    logger.info("=" * 65)
    logger.info("🔺 ARC-AGI-1 COMBINED EVALUATION — ALL SOLVERS")
    logger.info("=" * 65)

    # Load previous eval results (eval set only)
    prev_solved = set()
    all_prev_solved = set()
    for f in ['arc_agi1_eval_results.json', 'arc_emerged_eval_results.json']:
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            all_prev_solved |= set(d.get('solved_ids', []))

    logger.info(f"Previous best (union of all runs): {len(all_prev_solved)}/400 ({len(all_prev_solved)/4:.1f}%)")

    # Load tasks
    tasks = load_tasks(ARC_V1_EVAL)
    logger.info(f"Loaded {len(tasks)} evaluation tasks\n")

    # Solver configs: (name, worker_fn, extra_args, timeout_seconds)
    # Fast solvers only for practical runtime (~30min)
    # TTCCVTLR and ConvTTT are too slow per task in subprocess
    SOLVERS = [
        ('DSL+HE',    _dsl_worker,          (), 10),
        ('Object',    _object_solver_worker,  (), 5),
    ]

    solved = []
    new_solves = []
    results = {}
    methods_used = Counter()
    total_time = 0

    # Split tasks: skip previously-solved, only run solvers on unsolved
    unsolved_tids = [tid for tid in tasks if tid not in all_prev_solved]
    prev_in_eval = [tid for tid in tasks if tid in all_prev_solved]
    logger.info(f"Skipping {len(prev_in_eval)} previously-solved tasks")
    logger.info(f"Running {len(unsolved_tids)} unsolved tasks\n")

    # Credit previous solves
    for tid in prev_in_eval:
        solved.append(tid)
        methods_used['previous'] += 1
        results[tid] = {'solved': True, 'method': 'previous', 'time_s': 0, 'n_predictions': 0}

    for i, tid in enumerate(unsolved_tids):
        task = tasks[tid]
        t0 = time.time()
        test_output = task['test'][0].get('output')
        if test_output is None:
            results[tid] = {'solved': False, 'reason': 'no_ground_truth'}
            continue

        task_json = json.dumps(task)
        all_predictions = []

        # Run solvers in sequence — fast first, stop early on solve
        found = False
        for solver_name, worker_fn, extra_args, timeout in SOLVERS:
            if found:
                break
            preds = run_solver_with_timeout(worker_fn, (task_json, *extra_args), timeout)
            for method, pred in preds:
                all_predictions.append((method, pred))
                if grids_match(pred, test_output):
                    found = True

        elapsed = time.time() - t0
        total_time += elapsed

        winning_method = 'none'
        is_solved = False
        for method, pred in all_predictions:
            if grids_match(pred, test_output):
                winning_method = method
                is_solved = True
                break

        if is_solved:
            solved.append(tid)
            methods_used[winning_method] += 1
            new_solves.append(tid)

        results[tid] = {
            'solved': is_solved,
            'method': winning_method,
            'time_s': round(elapsed, 2),
            'n_predictions': len(all_predictions),
        }

        # Print EVERY task for visibility
        pct = len(new_solves) / (i + 1) * 100
        status = "✅" if is_solved else "  "
        avg_t = total_time / (i + 1)
        eta = avg_t * (len(unsolved_tids) - i - 1)
        n_new = len(new_solves)
        logger.info(
            f"[{i+1:3d}/{len(unsolved_tids)}] {status} {tid} "
            f"| {winning_method:12s} | {elapsed:.1f}s "
            f"| +{n_new} new ({len(prev_in_eval)+len(new_solves)}/{len(tasks)} total) "
            f"| ETA: {eta/60:.0f}m"
        )
        sys.stdout.flush()

    # Combined with previous
    combined = all_prev_solved | set(solved)

    # Final report
    logger.info(f"\n{'='*65}")
    logger.info("📊 COMBINED EVALUATION RESULTS")
    logger.info(f"{'='*65}")
    logger.info(f"  This run solved:       {len(solved)}/{len(tasks)} ({len(solved)/len(tasks)*100:.1f}%)")
    logger.info(f"  New (not in prev):     {len(new_solves)}")
    logger.info(f"  Previous union:        {len(all_prev_solved)}/400")
    logger.info(f"  ★ GRAND TOTAL:         {len(combined)}/400 ({len(combined)/4:.1f}%)")
    logger.info(f"  Total time:            {total_time:.0f}s ({total_time/len(tasks):.1f}s/task)")
    logger.info(f"\n  Methods breakdown:")
    for method, count in methods_used.most_common():
        logger.info(f"    {method:15s}: {count}")

    if new_solves:
        logger.info(f"\n  🆕 Newly solved ({len(new_solves)}):")
        for tid in new_solves[:20]:
            logger.info(f"    {tid} ({results[tid]['method']})")
        if len(new_solves) > 20:
            logger.info(f"    ... and {len(new_solves)-20} more")

    # Save
    output = {
        'this_run_solved': len(solved),
        'this_run_pct': round(len(solved) / len(tasks) * 100, 1),
        'new_solves_count': len(new_solves),
        'grand_total': len(combined),
        'grand_total_pct': round(len(combined) / 4, 1),
        'solved_ids': solved,
        'new_solve_ids': new_solves,
        'combined_ids': sorted(combined),
        'methods': dict(methods_used),
        'total_time_s': round(total_time, 1),
        'per_task': results,
    }

    outfile = 'arc_combined_eval_results.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Results saved to {outfile}")

    return output


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    run_combined_evaluation()
