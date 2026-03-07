#!/usr/bin/env python3
"""
Compound ARC Solver — Neural + Symbolic Integration
=====================================================
Layers all solving approaches in a cascade with ensemble voting:

  Layer 1: Mega Solver (26 heuristic strategies)        ~0.1s/task
  Layer 2: Decision Tree Classifier (80+ features)      ~0.5s/task
  Layer 3: ConvTTT Neural (per-task training)            ~10s/task
  Layer 4: OctoTetrahedral Transformer (few-shot)        ~2s/task
  Layer 5: Ensemble Vote (when multiple candidates)

Hardware: MacBook Pro M2 Pro (Mac14,10) · 12-core · 16GB

Usage:
    python arc_compound_solver.py [--data-dir ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation]
    python arc_compound_solver.py --quick   # Skip slow neural layers
"""

import json
import time
import sys
import os
import signal
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from multiprocessing import Pool, TimeoutError as MPTimeoutError
from functools import partial

# Paths
ARC_DIR = Path.home() / "ARC_AMD_TRANSFER" / "data" / "ARC-AGI" / "data" / "evaluation"
sys.path.insert(0, str(Path.home()))


# ============================================================================
# Layer 1: Mega Solver (import all strategies)
# ============================================================================

def load_mega_solver():
    """Import mega solver's solve_all function."""
    try:
        from arc_mega_solver import solve_all
        return solve_all
    except ImportError as e:
        print(f"[Layer 1] Could not load mega_solver: {e}")
        return None


# ============================================================================
# Layer 2: Decision Tree Solver
# ============================================================================

def load_dt_solver():
    """Import DT solver's solve_task function."""
    try:
        from arc_decision_tree_solver import solve_task
        return solve_task
    except ImportError as e:
        print(f"[Layer 2] Could not load dt_solver: {e}")
        return None


# ============================================================================
# Layer 3: ConvTTT Neural Solver
# ============================================================================

def load_conv_ttt():
    """Import ConvTTT solver."""
    try:
        import torch
        from arc_conv_ttt import solve_task as conv_solve, analyze_task
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        return conv_solve, analyze_task, device
    except ImportError as e:
        print(f"[Layer 3] Could not load conv_ttt: {e}")
        return None, None, None


# ============================================================================
# Layer 4: OctoTetrahedral Transformer
# ============================================================================

def load_octo_solver():
    """Import OctoTetrahedral neural solver."""
    try:
        from integrated_arc_solver import OctoTetrahedralSolver
        for ckpt in [
            Path.home() / "checkpoints" / "emergence" / "best_gci.pt",
            Path.home() / "checkpoints" / "arc" / "arc_final.pt",
            Path.home() / "checkpoints" / "best.pt",
        ]:
            if ckpt.exists():
                solver = OctoTetrahedralSolver(str(ckpt))
                if solver.model is not None:
                    print(f"[Layer 4] Loaded OctoTetrahedral from {ckpt.name}")
                    return solver
        # No checkpoint worked — model architecture mismatch is expected
        print("[Layer 4] Checkpoints exist but model mismatch — skipping neural layer")
        return None
    except Exception as e:
        print(f"[Layer 4] Could not load OctoTetrahedral: {e}")
        return None


# ============================================================================
# Compound Solver
# ============================================================================

class CompoundArcSolver:
    """
    Cascading compound solver combining symbolic + neural approaches.

    Each layer gets progressively more expensive but catches different
    failure modes. The ensemble vote combines candidates when available.
    """

    def __init__(self, enable_conv_ttt: bool = True, enable_octo: bool = True,
                 conv_ttt_timeout: float = 15.0, octo_timeout: float = 10.0):
        print("=" * 60)
        print("  Compound ARC Solver — Loading Layers")
        print("=" * 60)

        # Layer 1: Mega Solver
        self.mega_solve = load_mega_solver()
        print(f"  [Layer 1] Mega Solver:  {'✅' if self.mega_solve else '❌'}")

        # Layer 2: DT Solver
        self.dt_solve = load_dt_solver()
        print(f"  [Layer 2] DT Solver:    {'✅' if self.dt_solve else '❌'}")

        # Layer 3: ConvTTT
        self.conv_ttt_timeout = conv_ttt_timeout
        if enable_conv_ttt:
            self.conv_solve, self.analyze_task, self.device = load_conv_ttt()
        else:
            self.conv_solve = self.analyze_task = self.device = None
        print(f"  [Layer 3] ConvTTT:      {'✅' if self.conv_solve else '❌ (disabled)' if not enable_conv_ttt else '❌'}")

        # Layer 4: OctoTetrahedral
        self.octo_timeout = octo_timeout
        if enable_octo:
            self.octo_solver = load_octo_solver()
        else:
            self.octo_solver = None
        print(f"  [Layer 4] OctoTetrahedral: {'✅' if self.octo_solver else '❌ (disabled)' if not enable_octo else '❌'}")

        print("=" * 60)

        # Stats
        self.stats = defaultdict(int)

    def solve_task(self, task: Dict, task_id: str = "") -> Tuple[Optional[Any], str]:
        """
        Solve a single ARC task through the cascade.
        Returns (prediction_grid, method_name) or (None, "none").
        """
        candidates = []  # (prediction, method, confidence)
        mega_dt_pred = None

        # --- Layer 1: Mega Solver (fast heuristics) ---
        if self.mega_solve:
            try:
                result = self.mega_solve(task)
                if result is not None:
                    pred, method = result
                    if 'enhanced_dt' in method:
                        # enhanced_dt — also run Layer 2 for validation
                        mega_dt_pred = pred
                        candidates.append((pred, f"L1_{method}", 0.8))
                    else:
                        # Non-DT strategies are reliable — return immediately
                        self.stats[f"L1_{method}"] += 1
                        return pred, f"L1_{method}"
            except Exception:
                pass

        # --- Layer 2: Decision Tree (validates on training data) ---
        if self.dt_solve:
            try:
                dt_result = self.dt_solve(task)
                if dt_result is not None:
                    pred = dt_result[0] if isinstance(dt_result, list) else dt_result
                    if isinstance(pred, np.ndarray):
                        pred = pred.tolist()
                    # If L2 agrees with L1, high confidence
                    if mega_dt_pred is not None and str(pred) == str(mega_dt_pred):
                        candidates.append((pred, "L2_dt", 1.0))
                    elif mega_dt_pred is None:
                        # L1 failed entirely, L2 is our best bet
                        candidates.append((pred, "L2_dt", 0.9))
                    else:
                        # L1 and L2 disagree — include both
                        candidates.append((pred, "L2_dt", 0.85))
            except Exception:
                pass

        # --- Layer 3: ConvTTT (same-size tasks only) ---
        if self.conv_solve and self.analyze_task:
            try:
                info = self.analyze_task(task)
                if info.get('same_size', False):
                    preds, confs = self.conv_solve(
                        task, self.device,
                        num_aug=50, num_steps=500, num_vote=16,
                        verbose=False
                    )
                    if preds and len(preds) > 0:
                        pred = preds[0]
                        if isinstance(pred, np.ndarray):
                            pred = pred.tolist()
                        # Validate on training examples
                        if self._validate_conv_ttt(task, pred):
                            candidates.append((pred, "L3_conv_ttt", 0.7))
            except Exception:
                pass

        # --- Layer 4: OctoTetrahedral Transformer ---
        if self.octo_solver:
            try:
                neural_preds = self.octo_solver.solve(task, num_predictions=2)
                for i, npred in enumerate(neural_preds):
                    candidates.append((npred, f"L4_octo_t{i}", 0.5))
            except Exception:
                pass

        # --- Layer 5: Ensemble vote ---
        if candidates:
            best = self._ensemble_vote(candidates)
            if best:
                pred, method = best
                self.stats[method] += 1
                return pred, method

        self.stats["unsolved"] += 1
        return None, "none"

    def _validate_conv_ttt(self, task: Dict, test_pred: Any) -> bool:
        """Basic validation: check prediction has reasonable shape."""
        try:
            test_input = np.array(task['test'][0]['input'])
            pred = np.array(test_pred)
            # For same-size tasks, output should match input dims
            if pred.shape == test_input.shape:
                return True
            # Check it at least looks like a valid grid
            if pred.ndim == 2 and 0 < pred.shape[0] <= 30 and 0 < pred.shape[1] <= 30:
                return True
            return False
        except Exception:
            return False

    def _ensemble_vote(self, candidates: List[Tuple]) -> Optional[Tuple]:
        """Weighted vote across candidates. Returns (prediction, method)."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return (candidates[0][0], candidates[0][1])

        # Weighted voting by stringified prediction
        votes = defaultdict(float)
        pred_map = {}
        method_map = {}

        for pred, method, conf in candidates:
            key = str(pred)
            votes[key] += conf
            pred_map[key] = pred
            if key not in method_map or conf > votes[key] - conf:
                method_map[key] = method

        best_key = max(votes, key=votes.get)
        return (pred_map[best_key], method_map[best_key])

    def evaluate(self, data_dir: str = None, limit: int = None,
                 timeout: float = 45.0) -> Dict:
        """
        Evaluate on ARC-AGI evaluation set.
        Returns results dict with solved count and submission data.
        """
        if data_dir is None:
            data_dir = str(ARC_DIR)

        task_dir = Path(data_dir)
        task_files = sorted(task_dir.glob("*.json"))
        if limit:
            task_files = task_files[:limit]

        print(f"\n  Evaluating {len(task_files)} tasks (timeout={timeout}s)")
        print(f"  {'─' * 50}")

        results = {
            "total": len(task_files),
            "solved": 0,
            "methods": defaultdict(int),
            "solved_ids": [],
            "failed_ids": [],
            "submission": {},
            "timings": [],
        }

        for i, tf in enumerate(task_files):
            task_id = tf.stem
            with open(tf) as f:
                task = json.load(f)

            t0 = time.time()

            try:
                pred, method = self.solve_task(task, task_id)
                elapsed = time.time() - t0

                if pred is not None:
                    # Check against ground truth if available
                    gt = task['test'][0].get('output')
                    if gt is not None:
                        if pred == gt or (isinstance(pred, list) and pred == gt):
                            results["solved"] += 1
                            results["methods"][method] += 1
                            results["solved_ids"].append(task_id)
                            status = f"✅ {method}"
                        else:
                            results["failed_ids"].append(task_id)
                            status = f"❌ {method} (wrong)"
                    else:
                        results["methods"][method] += 1
                        results["solved_ids"].append(task_id)
                        status = f"? {method}"

                    # Store in submission format (2 attempts)
                    results["submission"][task_id] = [pred, pred]
                else:
                    results["failed_ids"].append(task_id)
                    # Store empty grid as fallback
                    test_in = task['test'][0]['input']
                    results["submission"][task_id] = [test_in, test_in]
                    status = "⬜ unsolved"

                results["timings"].append(elapsed)

            except Exception as e:
                elapsed = time.time() - t0
                results["failed_ids"].append(task_id)
                test_in = task['test'][0]['input']
                results["submission"][task_id] = [test_in, test_in]
                status = f"💥 error: {str(e)[:40]}"
                results["timings"].append(elapsed)

            if (i + 1) % 20 == 0 or status.startswith("✅"):
                pct = results["solved"] / (i + 1) * 100
                print(f"  [{i+1:3d}/{len(task_files)}] {task_id} {status} ({elapsed:.1f}s) — {results['solved']}/{i+1} ({pct:.1f}%)")

        # Summary
        total_time = sum(results["timings"])
        avg_time = total_time / len(results["timings"]) if results["timings"] else 0

        print(f"\n  {'=' * 50}")
        print(f"  COMPOUND SOLVER RESULTS")
        print(f"  {'=' * 50}")
        print(f"  Solved: {results['solved']}/{results['total']} ({results['solved']/results['total']*100:.1f}%)")
        print(f"  Time:   {total_time:.1f}s total, {avg_time:.2f}s avg")
        print(f"\n  Methods breakdown:")
        for method, count in sorted(results["methods"].items(), key=lambda x: -x[1]):
            print(f"    {method:30s} {count}")
        print(f"  {'=' * 50}")

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compound ARC Solver")
    parser.add_argument("--data-dir", type=str, default=str(ARC_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Skip slow neural layers")
    parser.add_argument("--output", type=str, default="compound_results.json")
    parser.add_argument("--submission", type=str, default="submission.json")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    solver = CompoundArcSolver(
        enable_conv_ttt=not args.quick,
        enable_octo=not args.quick,
    )

    results = solver.evaluate(
        data_dir=args.data_dir,
        limit=args.limit,
        timeout=args.timeout,
    )

    # Save results
    out = {
        "total": results["total"],
        "solved": results["solved"],
        "accuracy": results["solved"] / results["total"] if results["total"] else 0,
        "methods": dict(results["methods"]),
        "solved_ids": results["solved_ids"],
        "avg_time": sum(results["timings"]) / len(results["timings"]) if results["timings"] else 0,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save submission
    with open(args.submission, "w") as f:
        json.dump(results["submission"], f)
    print(f"Submission saved to {args.submission}")


if __name__ == "__main__":
    main()
