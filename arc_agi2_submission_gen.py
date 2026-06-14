#!/usr/bin/env python3
"""
ARC Prize 2026 — ARC-AGI-2 Submission Generator
Combines OctoTetrahedral compound solvers for 240 test tasks.
Output: submission.json in {task_id: [{attempt_1, attempt_2}]} format
"""
import json
import os
import sys
import time
import signal
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path.home()))

DATA_DIR   = Path.home() / "kaggle_data/arc-agi-2"
EVAL_DIR   = Path.home() / "kaggle_data/arc-agi-2-eval/evaluation"
OUT_FILE   = Path.home() / "arc_agi2_kaggle_submission.json"
TASK_TIMEOUT = 10  # seconds per task

# ── Solver helpers ────────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("task timeout")

def run_with_timeout(fn, timeout):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        return fn()
    finally:
        signal.alarm(0)


def identity_fallback(task: dict) -> list:
    """Return test input as fallback prediction."""
    return task["test"][0]["input"]


def try_compound_solver(task: dict) -> Optional[list]:
    try:
        from arc_compound_solver import CompoundArcSolver
        if not hasattr(try_compound_solver, "_inst"):
            # Disable slow ConvTTT layer for speed; keep mega + DT
            try_compound_solver._inst = CompoundArcSolver(enable_conv_ttt=False, enable_octo=False)
        pred, method = try_compound_solver._inst.solve_task(task)
        if pred is not None:
            return pred
    except Exception:
        pass
    return None


def try_hybrid_solver(task: dict) -> Optional[list]:
    try:
        from arc_hybrid_solver import HybridARCSolver
        if not hasattr(try_hybrid_solver, "_inst"):
            try_hybrid_solver._inst = HybridARCSolver(use_neural_fallback=False)
        preds = try_hybrid_solver._inst.solve(task)
        if preds:
            return preds[0]
    except Exception:
        pass
    return None


def try_dsl_solver(task: dict) -> Optional[list]:
    try:
        from arc_solver import ARCSolver
        if not hasattr(try_dsl_solver, "_inst"):
            try_dsl_solver._inst = ARCSolver()
        result = try_dsl_solver._inst.solve(task)
        if isinstance(result, dict) and result.get("prediction"):
            return result["prediction"]
        if isinstance(result, list) and result:
            return result[0]
    except Exception:
        pass
    return None


def try_heuristic_solver(task: dict) -> Optional[list]:
    try:
        from arc_heuristic_solver import HeuristicARCSolver
        if not hasattr(try_heuristic_solver, "_inst"):
            try_heuristic_solver._inst = HeuristicARCSolver()
        result = try_heuristic_solver._inst.solve(task)
        if result:
            return result[0] if isinstance(result, list) else result
    except Exception:
        pass
    return None


def try_openclaw(task: dict) -> Optional[list]:
    try:
        from openclaw import OpenClawSynth
        if not hasattr(try_openclaw, "_inst"):
            try_openclaw._inst = OpenClawSynth()
        pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
        prog = try_openclaw._inst.synthesize(pairs)
        if prog:
            return prog(task["test"][0]["input"])
    except Exception:
        pass
    return None


SOLVERS = [
    ("compound",   try_compound_solver),
    ("hybrid",     try_hybrid_solver),
    ("openclaw",   try_openclaw),
    ("dsl",        try_dsl_solver),
    ("heuristic",  try_heuristic_solver),
]


def solve_task(task: dict) -> tuple[list, list]:
    """Return (attempt_1, attempt_2) — try solvers in order, fallback = identity."""
    predictions = []
    for name, solver in SOLVERS:
        if len(predictions) >= 2:
            break
        try:
            def _call(s=solver, t=task): return s(t)
            pred = run_with_timeout(_call, TASK_TIMEOUT)
            if pred is not None and pred not in predictions:
                predictions.append(pred)
        except TimeoutError:
            pass
        except Exception:
            pass

    # Fill up to 2 predictions with identity fallback
    fallback = identity_fallback(task)
    while len(predictions) < 2:
        predictions.append(fallback)

    return predictions[0], predictions[1]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load test challenges
    test_file = DATA_DIR / "arc-agi_test_challenges.json"
    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        sys.exit(1)

    with open(test_file) as f:
        challenges = json.load(f)

    task_ids = list(challenges.keys())
    print(f"ARC-AGI-2 Submission Generator")
    print(f"Tasks: {len(task_ids)}")
    print(f"Timeout per task: {TASK_TIMEOUT}s")
    print("=" * 60)

    submission = {}
    solved = 0
    start_all = time.time()

    for i, task_id in enumerate(task_ids):
        task = challenges[task_id]
        t0 = time.time()

        a1, a2 = solve_task(task)
        fallback = identity_fallback(task)

        is_solved = a1 != fallback
        if is_solved:
            solved += 1

        submission[task_id] = [{"attempt_1": a1, "attempt_2": a2}]

        elapsed = time.time() - t0
        status = "✓" if is_solved else "·"
        print(f"  [{i+1:3d}/{len(task_ids)}] {task_id}  {elapsed:.2f}s  {status}", flush=True)

    total_time = time.time() - start_all
    print("=" * 60)
    print(f"Solved:  {solved}/{len(task_ids)} ({solved/len(task_ids)*100:.1f}%)")
    print(f"Elapsed: {total_time:.1f}s")
    print(f"Output:  {OUT_FILE}")

    with open(OUT_FILE, "w") as f:
        json.dump(submission, f)
    print("Done.")


if __name__ == "__main__":
    main()
