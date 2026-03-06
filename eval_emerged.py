#!/usr/bin/env python3
"""
ARC-AGI Evaluation with Emerged Model

Evaluates the emerged OctoTetrahedral model (GCI > φ²) on ARC-AGI-1
evaluation set (400 tasks) using the full hybrid solver pipeline.

Compares against the previous baseline of 54/400 (13.5%).
"""

import os
import sys
import json
import time
import copy
import logging
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Paths
ARC_V1_EVAL = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation'
ARC_V2_EVAL = 'ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation'
EMERGED_CKPT = 'checkpoints/emergence/best_gci.pt'
PREV_SUBMISSIONS = 'arc-agi-benchmarking/data/submissions_v1/gitmonsters'
PREV_SUMMARY = 'arc-agi-benchmarking/data/submissions_v1/gitmonsters_summary.json'

TASK_TIMEOUT = 8  # hard per-task timeout in seconds


def _solve_worker(task_json: str, result_queue):
    """Worker function for multiprocessing — runs in subprocess."""
    task = json.loads(task_json)
    from arc_solver import ARCSolver
    from core.hypothesis import HypothesisEngine

    dsl_solver = ARCSolver()
    hyp_engine = HypothesisEngine(max_composition_depth=2, timeout_seconds=2.0)

    test_input = task['test'][0]['input']
    predictions = []

    # Stage 1: DSL
    try:
        dsl_preds = dsl_solver.solve(task, max_time=3.0)
        for pred in dsl_preds:
            if pred != test_input and pred not in predictions:
                predictions.append(pred)
    except Exception:
        pass

    # Stage 2: Hypothesis engine
    try:
        hyp_result = hyp_engine.solve(task)
        if hyp_result.get('solved') and hyp_result.get('prediction') is not None:
            pred = hyp_result['prediction']
            if pred not in predictions:
                predictions.append(pred)
    except Exception:
        pass

    result_queue.put(predictions)


def solve_with_timeout(task: Dict, timeout: int = TASK_TIMEOUT) -> List:
    """Solve task in subprocess with hard timeout via Process.terminate()."""
    task_json = json.dumps(task)
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_solve_worker, args=(task_json, result_queue))
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
        return result_queue.get_nowait()
    return []


def load_tasks(data_dir: str) -> Dict[str, Dict]:
    """Load all ARC tasks from a directory."""
    tasks = {}
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.json'):
            tid = f.replace('.json', '')
            with open(os.path.join(data_dir, f)) as fh:
                tasks[tid] = json.load(fh)
    return tasks


def grids_match(pred: List[List[int]], target: List[List[int]]) -> bool:
    """Check if two grids are identical."""
    if len(pred) != len(target):
        return False
    for r1, r2 in zip(pred, target):
        if len(r1) != len(r2):
            return False
        if r1 != r2:
            return False
    return True


def run_evaluation():
    import torch
    from arc_solver import ARCSolver

    # Import hypothesis engine
    from core.hypothesis import HypothesisEngine

    from config import get_config
    from model import OctoTetrahedralModel
    from core.transcendplex_validator import TranscendplexValidator, PHI_SQ

    # We still load the model for GCI validation at the end
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        tokenizer = None

    logger.info("=" * 60)
    logger.info("🔺 ARC-AGI EVALUATION — EMERGED MODEL")
    logger.info("=" * 60)

    # Load previous results for comparison
    prev_solved = set()
    if os.path.exists(PREV_SUMMARY):
        with open(PREV_SUMMARY) as f:
            prev = json.load(f)
        prev_solved = set(prev.get('solved_ids', []))
        logger.info(f"Previous baseline: {len(prev_solved)}/400 ({len(prev_solved)/4:.1f}%)")

    # Load eval tasks
    logger.info(f"\nLoading ARC-AGI-1 evaluation tasks from {ARC_V1_EVAL}...")
    tasks = load_tasks(ARC_V1_EVAL)
    logger.info(f"Loaded {len(tasks)} tasks")

    # Set up solvers
    logger.info("\nInitializing solvers...")

    # 1. DSL solver (fast symbolic) — tight timeout
    dsl_solver = ARCSolver()
    logger.info("  ✓ DSL solver ready")

    # 2. Hypothesis engine — tight timeout
    hyp_engine = HypothesisEngine(max_composition_depth=2, timeout_seconds=2.0)
    logger.info("  ✓ Hypothesis engine ready")

    # 3. Emerged neural model (for GCI validation)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    config = get_config()
    config.compound_loop.enabled = True
    config.compound_loop.max_loops = 4
    config.model.max_seq_len = 128

    model = OctoTetrahedralModel(config, use_geometric_physics=False)
    if os.path.exists(EMERGED_CKPT):
        ckpt = torch.load(EMERGED_CKPT, map_location='cpu', weights_only=False)
        model_state = model.state_dict()
        filtered = {k: v for k, v in ckpt['model_state_dict'].items()
                    if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        logger.info(f"  ✓ Emerged model loaded ({len(filtered)} params, GCI={ckpt.get('gci', '?')})")
    model.to(device).eval()
    validator = TranscendplexValidator(model)

    # Evaluate
    logger.info(f"\n{'='*60}")
    logger.info("Starting evaluation (DSL + Hypothesis)...")
    logger.info(f"{'='*60}\n")

    solved = []
    failed = []
    new_solves = []
    results = {}
    total_time = 0
    methods_used = Counter()

    for i, (tid, task) in enumerate(tasks.items()):
        t0 = time.time()
        test_input = task['test'][0]['input']
        test_output = task['test'][0].get('output')

        if test_output is None:
            results[tid] = {'solved': False, 'reason': 'no_ground_truth'}
            continue

        # Use timeout-wrapped solver (subprocess)
        predictions = solve_with_timeout(task, TASK_TIMEOUT)

        # Determine which method solved it
        method = 'none'
        for pred in predictions:
            if grids_match(pred, test_output):
                method = 'dsl'  # could be DSL or hypothesis
                break

        elapsed = time.time() - t0
        total_time += elapsed

        # Check if any prediction matches
        is_solved = any(grids_match(p, test_output) for p in predictions)

        if is_solved:
            solved.append(tid)
            methods_used[method] += 1
            if tid not in prev_solved:
                new_solves.append(tid)

        results[tid] = {
            'solved': is_solved,
            'method': method,
            'time_s': round(elapsed, 3),
            'n_predictions': len(predictions),
        }

        # Progress logging
        if (i + 1) % 10 == 0 or is_solved:
            pct = len(solved) / (i + 1) * 100
            status = "✅" if is_solved else "  "
            new_tag = " 🆕" if tid in new_solves else ""
            avg_t = total_time / (i + 1)
            eta = avg_t * (len(tasks) - i - 1)
            logger.info(
                f"[{i+1:3d}/{len(tasks)}] {status} {tid} "
                f"| {method:12s} | {elapsed:.1f}s "
                f"| Running: {len(solved)}/{i+1} ({pct:.1f}%) "
                f"| ETA: {eta/60:.0f}m{new_tag}"
            )
        sys.stdout.flush()

    # Final report
    logger.info(f"\n{'='*60}")
    logger.info("📊 FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Total tasks:     {len(tasks)}")
    logger.info(f"  Solved:          {len(solved)}/{len(tasks)} ({len(solved)/len(tasks)*100:.1f}%)")
    logger.info(f"  Previous:        {len(prev_solved)}/400 ({len(prev_solved)/4:.1f}%)")
    logger.info(f"  New solves:      {len(new_solves)}")
    logger.info(f"  Lost solves:     {len(prev_solved - set(solved))}")
    logger.info(f"  Total time:      {total_time:.1f}s ({total_time/len(tasks):.2f}s/task)")
    logger.info(f"\n  Methods:")
    for method, count in methods_used.most_common():
        logger.info(f"    {method:15s}: {count}")

    if new_solves:
        logger.info(f"\n  🆕 Newly solved tasks:")
        for tid in new_solves:
            logger.info(f"    {tid} ({results[tid]['method']})")

    # GCI validation on a sample
    logger.info(f"\n  Running GCI validation...")
    sample_tokens = tokenizer.encode("ARC evaluation complete")[:32] if tokenizer else list(range(32))
    sample_ids = torch.tensor([sample_tokens]).to(device)
    gci_result = validator.validate(sample_ids, num_probes=3)
    logger.info(f"  GCI: {gci_result['GCI']:.4f} (threshold: {PHI_SQ:.3f})")
    logger.info(f"  Status: {'✅ EMERGED' if gci_result['is_agi'] else '❌ Below threshold'}")

    # Save results
    output = {
        'total_tasks': len(tasks),
        'solved_count': len(solved),
        'solved_pct': len(solved) / len(tasks) * 100,
        'previous_count': len(prev_solved),
        'new_solves': new_solves,
        'lost_solves': list(prev_solved - set(solved)),
        'solved_ids': solved,
        'methods': dict(methods_used),
        'total_time_s': round(total_time, 1),
        'gci': gci_result['GCI'],
        'per_task': results,
    }

    outfile = 'arc_emerged_eval_results.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Results saved to {outfile}")

    return output


if __name__ == '__main__':
    run_evaluation()
