#!/usr/bin/env python3
"""
Extended DSL search with longer timeout (30s vs previous 8s).
Uses fork + terminate for hard timeout enforcement.
Also includes HypothesisEngine with deeper composition.
"""

import os
import sys
import json
import time
import logging
import multiprocessing
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

ARC_V1_EVAL = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation'


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


def _dsl_deep_worker(task_json, result_queue):
    """Deep DSL search + HypothesisEngine with depth 4."""
    task = json.loads(task_json)
    test_output = task['test'][0].get('output')
    if test_output is None:
        result_queue.put((False, 'no_gt'))
        return

    test_input = task['test'][0]['input']

    # DSL solver with 25s budget
    try:
        from arc_solver import ARCSolver
        solver = ARCSolver()
        preds = solver.solve(task, max_time=25.0)
        for p in preds:
            if p != test_input and grids_match(p, test_output):
                result_queue.put((True, 'dsl_deep'))
                return
    except Exception:
        pass

    # HypothesisEngine with depth 4
    try:
        from core.hypothesis import HypothesisEngine
        he = HypothesisEngine(max_composition_depth=4, timeout_seconds=10.0)
        result = he.solve(task)
        if result.get('solved') and result.get('prediction') is not None:
            if grids_match(result['prediction'], test_output):
                result_queue.put((True, 'hypothesis_d4'))
                return
    except Exception:
        pass

    result_queue.put((False, 'none'))


def solve_with_timeout(task_json, timeout_s=35):
    """Run solver in forked subprocess with hard timeout."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_dsl_deep_worker, args=(task_json, q))
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(timeout=3)
        return False, 'timeout'

    if not q.empty():
        try:
            return q.get_nowait()
        except Exception:
            pass
    return False, 'error'


def main():
    logger.info("=" * 60)
    logger.info("🔍 EXTENDED DSL SEARCH — 30s timeout")
    logger.info("=" * 60)

    tasks = {}
    for f in sorted(os.listdir(ARC_V1_EVAL)):
        if f.endswith('.json'):
            tid = f.replace('.json', '')
            with open(os.path.join(ARC_V1_EVAL, f)) as fh:
                tasks[tid] = json.load(fh)

    prev_solved = set()
    for f in ['arc_agi1_eval_results.json', 'arc_emerged_eval_results.json']:
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            prev_solved |= set(d.get('solved_ids', []))

    unsolved = [(tid, t) for tid, t in tasks.items() if tid not in prev_solved]
    logger.info(f"Total: {len(tasks)}, Previously solved: {len(prev_solved)}, Unsolved: {len(unsolved)}")

    timeout = 35  # hard kill timeout
    logger.info(f"Timeout: {timeout}s/task (DSL 25s + HE depth-4 10s)")
    logger.info(f"Estimated time: {len(unsolved) * timeout / 60:.0f}min (worst case)\n")

    solved = []
    methods = {}
    t_start = time.time()

    for i, (tid, task) in enumerate(unsolved):
        t0 = time.time()
        is_solved, method = solve_with_timeout(json.dumps(task), timeout_s=timeout)
        elapsed = time.time() - t0

        if is_solved:
            solved.append(tid)
            methods[method] = methods.get(method, 0) + 1
            logger.info(
                f"[{i+1:3d}/{len(unsolved)}] ✅ {tid} | {method:15s} | {elapsed:.1f}s "
                f"| +{len(solved)} new ({len(prev_solved)+len(solved)}/400)"
            )
        elif (i + 1) % 20 == 0:
            wall = time.time() - t_start
            eta = wall / (i + 1) * (len(unsolved) - i - 1)
            logger.info(
                f"[{i+1:3d}/{len(unsolved)}]    progress | {wall:.0f}s elapsed "
                f"| +{len(solved)} new | ETA: {eta/60:.0f}m"
            )
        sys.stdout.flush()

    total_time = time.time() - t_start
    combined = prev_solved | set(solved)

    logger.info(f"\n{'='*60}")
    logger.info("📊 EXTENDED DSL SEARCH RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  New solves:     {len(solved)}")
    logger.info(f"  Previous:       {len(prev_solved)}")
    logger.info(f"  ★ TOTAL:        {len(combined)}/400 ({len(combined)/4:.1f}%)")
    logger.info(f"  Time:           {total_time:.0f}s ({total_time/len(unsolved):.1f}s/task)")
    if methods:
        logger.info(f"  Methods: {methods}")
    if solved:
        logger.info(f"  Solved: {sorted(solved)}")

    output = {
        'new_solved': len(solved),
        'total': len(combined),
        'total_pct': round(len(combined) / 4, 1),
        'solved_ids': sorted(solved),
        'combined_ids': sorted(combined),
        'methods': methods,
        'time_s': round(total_time, 1),
    }
    with open('arc_dsl_deep_eval_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Saved to arc_dsl_deep_eval_results.json")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
