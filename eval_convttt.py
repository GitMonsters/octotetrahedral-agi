#!/usr/bin/env python3
"""
ConvTTT evaluation on ARC-AGI-1 eval set.
Uses fork + terminate for hard timeout. Runs on same-size tasks only.
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


def _conv_ttt_worker(task_json, n_seeds, result_queue):
    """ConvTTT worker in forked subprocess."""
    import torch
    task = json.loads(task_json)
    test_output = task['test'][0].get('output')
    if test_output is None:
        result_queue.put((False, 'no_gt', []))
        return

    try:
        from arc_conv_ttt import solve_task as ttt_solve
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

        best_pred = None
        best_score = -1

        for seed in range(n_seeds):
            torch.manual_seed(seed * 42 + 7)
            np.random.seed(seed * 42 + 7)
            preds = ttt_solve(task, device=device, verbose=False)
            
            if preds is None:
                continue

            for p in preds:
                if isinstance(p, torch.Tensor):
                    p = p.cpu().numpy()
                p = np.array(p)
                p = np.squeeze(p)
                
                if p.ndim != 2:
                    continue

                p_list = p.astype(int).tolist()
                if grids_match(p_list, test_output):
                    result_queue.put((True, f'conv_ttt_s{seed}', [p_list]))
                    return

                # Track best partial match
                target = np.array(test_output)
                if p.shape == target.shape:
                    score = np.sum(p.astype(int) == target)
                    if score > best_score:
                        best_score = score
                        best_pred = p_list

        result_queue.put((False, 'none', [best_pred] if best_pred else []))
    except Exception as e:
        result_queue.put((False, f'error:{e}', []))


def solve_with_timeout(task_json, n_seeds, timeout_s):
    """Run ConvTTT in forked subprocess with hard timeout."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_conv_ttt_worker, args=(task_json, n_seeds, q))
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(timeout=3)
        return False, 'timeout', []

    if not q.empty():
        try:
            return q.get_nowait()
        except Exception:
            pass
    return False, 'error', []


def main():
    logger.info("=" * 60)
    logger.info("🧠 ConvTTT EVALUATION — ARC-AGI-1 EVAL SET")
    logger.info("=" * 60)

    # Load tasks
    tasks = {}
    for f in sorted(os.listdir(ARC_V1_EVAL)):
        if f.endswith('.json'):
            tid = f.replace('.json', '')
            with open(os.path.join(ARC_V1_EVAL, f)) as fh:
                tasks[tid] = json.load(fh)

    # Load previously solved
    prev_solved = set()
    for f in ['arc_agi1_eval_results.json', 'arc_emerged_eval_results.json']:
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            prev_solved |= set(d.get('solved_ids', []))

    # Filter to same-size unsolved tasks
    candidates = []
    for tid, task in tasks.items():
        if tid in prev_solved:
            continue
        # Check if all examples are same-size
        all_same = True
        for ex in task['train']:
            if np.array(ex['input']).shape != np.array(ex['output']).shape:
                all_same = False
                break
        test_in = np.array(task['test'][0]['input'])
        test_out = task['test'][0].get('output')
        if test_out is not None:
            if test_in.shape != np.array(test_out).shape:
                all_same = False
        if all_same:
            candidates.append((tid, task))

    logger.info(f"Total eval tasks: {len(tasks)}")
    logger.info(f"Previously solved: {len(prev_solved)}")
    logger.info(f"Same-size unsolved candidates: {len(candidates)}")

    n_seeds = 4
    timeout = 50  # seconds per task

    logger.info(f"Seeds: {n_seeds}, Timeout: {timeout}s/task")
    logger.info(f"Estimated time: {len(candidates) * timeout / 60:.0f}min (worst case)\n")

    solved = []
    methods = {}
    t_start = time.time()

    for i, (tid, task) in enumerate(candidates):
        t0 = time.time()
        is_solved, method, preds = solve_with_timeout(json.dumps(task), n_seeds, timeout)
        elapsed = time.time() - t0

        if is_solved:
            solved.append(tid)
            methods[method] = methods.get(method, 0) + 1
            logger.info(
                f"[{i+1:3d}/{len(candidates)}] ✅ {tid} | {method:15s} | {elapsed:.1f}s "
                f"| +{len(solved)} new ({len(prev_solved)+len(solved)}/400)"
            )
        elif (i + 1) % 10 == 0:
            wall = time.time() - t_start
            eta = wall / (i + 1) * (len(candidates) - i - 1)
            logger.info(
                f"[{i+1:3d}/{len(candidates)}]    progress | {wall:.0f}s elapsed "
                f"| +{len(solved)} new | ETA: {eta/60:.0f}m"
            )
        sys.stdout.flush()

    total_time = time.time() - t_start
    combined = prev_solved | set(solved)

    logger.info(f"\n{'='*60}")
    logger.info("📊 ConvTTT EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  New solves:     {len(solved)}")
    logger.info(f"  Previous:       {len(prev_solved)}")
    logger.info(f"  ★ TOTAL:        {len(combined)}/400 ({len(combined)/4:.1f}%)")
    logger.info(f"  Time:           {total_time:.0f}s ({total_time/len(candidates):.1f}s/task)")
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
        'candidates_tested': len(candidates),
    }
    with open('arc_convttt_eval_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Saved to arc_convttt_eval_results.json")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
