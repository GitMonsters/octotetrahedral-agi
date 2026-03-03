#!/usr/bin/env python3
"""Combined ARC-AGI-2 evaluation pipeline.

Cascading solver chain:
1. Symbolic solver (arc_solver.py) — fast, exact when it works
2. Program synthesis (arc_synth.py) — general primitives + composition
3. ConvNet TTT (arc_conv_ttt.py) — per-task neural predictions
4. Brute-force search around ConvNet predictions

Returns 2 guesses per test input for the ARC Prize 2-attempt format.
"""
import os
import sys
import json
import time
import copy
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch


def load_tasks(data_dir: str) -> Dict[str, dict]:
    """Load all ARC tasks from directory."""
    tasks = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".json"):
            task_id = fname[:-5]
            with open(os.path.join(data_dir, fname)) as f:
                tasks[task_id] = json.load(f)
    return tasks


def validate_on_training(pred_fn, task: dict) -> bool:
    """Check if a prediction function works on all training examples."""
    for pair in task["train"]:
        inp = np.array(pair["input"])
        expected = np.array(pair["output"])
        pred = pred_fn(inp)
        if pred is None or pred.shape != expected.shape or not np.array_equal(pred, expected):
            return False
    return True


class CombinedSolver:
    def __init__(self, device: torch.device, verbose: bool = True):
        self.device = device
        self.verbose = verbose
        self._symbolic_solver = None
        self._synth_solver = None
    
    def _get_symbolic_solver(self):
        """Lazy-load symbolic solver."""
        if self._symbolic_solver is None:
            try:
                sys.path.insert(0, os.path.expanduser("~"))
                from arc_solver import ARCSolver
                self._symbolic_solver = ARCSolver()
                if self.verbose:
                    print("  Loaded symbolic solver (arc_solver.py)")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not load symbolic solver: {e}")
                self._symbolic_solver = False
        return self._symbolic_solver if self._symbolic_solver else None
    
    def _get_synth_solver(self):
        """Lazy-load synthesis solver."""
        if self._synth_solver is None:
            try:
                sys.path.insert(0, os.path.expanduser("~"))
                from arc_synth import ARCSynthesizer
                self._synth_solver = ARCSynthesizer()
                if self.verbose:
                    print("  Loaded synthesis solver (arc_synth.py)")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not load synthesis solver: {e}")
                self._synth_solver = False
        return self._synth_solver if self._synth_solver else None
    
    def solve_pattern(self, task: dict) -> List[Optional[np.ndarray]]:
        """Try pattern matching solver."""
        try:
            sys.path.insert(0, os.path.expanduser("~"))
            from arc_pattern_solver import solve_task as pattern_solve
            return pattern_solve(task)
        except Exception as e:
            if self.verbose:
                print(f"  Pattern error: {e}")
            return [None] * len(task["test"])
    
    def solve_symbolic(self, task: dict) -> List[Optional[np.ndarray]]:
        """Try symbolic solver."""
        solver = self._get_symbolic_solver()
        if solver is None:
            return [None] * len(task["test"])
        
        results = []
        for test_idx, test_pair in enumerate(task["test"]):
            try:
                # Normalize task for this test index
                task_copy = copy.deepcopy(task)
                if test_idx > 0:
                    task_copy["test"] = [task_copy["test"][test_idx]]
                
                pred = solver.solve(task_copy)
                if pred is not None:
                    pred = np.array(pred)
                    # Squeeze extra dimensions (solver may return [[grid]])
                    while pred.ndim > 2 and pred.shape[0] == 1:
                        pred = pred[0]
                    if pred.ndim != 2:
                        pred = None
                    else:
                        # Check it's not just the input
                        test_inp = np.array(test_pair["input"])
                        if pred.shape == test_inp.shape and np.array_equal(pred, test_inp):
                            pred = None  # Solver returned input unchanged
                results.append(pred)
            except Exception:
                results.append(None)
        
        return results
    
    def solve_synthesis(self, task: dict) -> List[Optional[np.ndarray]]:
        """Try program synthesis."""
        solver = self._get_synth_solver()
        if solver is None:
            return [None] * len(task["test"])
        
        results = []
        for test_idx, test_pair in enumerate(task["test"]):
            try:
                preds = solver.synthesize(task, test_idx=test_idx)
                if preds and len(preds) > 0:
                    pred = np.array(preds[0])
                    test_inp = np.array(test_pair["input"])
                    if pred.shape == test_inp.shape and np.array_equal(pred, test_inp):
                        pred = None
                    results.append(pred)
                else:
                    results.append(None)
            except Exception:
                results.append(None)
        
        return results
    
    def solve_convnet(self, task: dict, num_aug: int = 100, num_steps: int = 1000,
                      num_vote: int = 32) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """Try ConvNet test-time training. Returns (predictions, confidences)."""
        try:
            from arc_conv_ttt import solve_task
            predictions, confidences = solve_task(
                task, self.device,
                num_aug=num_aug,
                num_steps=num_steps,
                num_vote=num_vote,
                verbose=self.verbose,
            )
            return predictions, confidences
        except Exception as e:
            if self.verbose:
                print(f"  ConvNet error: {e}")
            return [None] * len(task["test"]), [None] * len(task["test"])
    
    def solve(self, task_id: str, task: dict,
             num_aug: int = 100, num_steps: int = 1000, num_vote: int = 32,
             ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Solve task, returning up to 2 guesses per test input.
        
        Returns: list of (guess1, guess2) tuples, one per test input.
        """
        num_tests = len(task["test"])
        guesses = [(None, None)] * num_tests
        
        # Phase 0: Pattern matching solver (very fast)
        t0 = time.time()
        pattern_preds = self.solve_pattern(task)
        pat_time = time.time() - t0
        solved_by_pat = sum(1 for p in pattern_preds if p is not None)
        if self.verbose and solved_by_pat:
            print(f"  Pattern: {solved_by_pat}/{num_tests} predictions ({pat_time:.1f}s)")
        
        # Phase 1: Symbolic solver (fast)
        t0 = time.time()
        symbolic_preds = self.solve_symbolic(task)
        sym_time = time.time() - t0
        
        solved_by_sym = sum(1 for p in symbolic_preds if p is not None)
        if self.verbose and solved_by_sym:
            print(f"  Symbolic: {solved_by_sym}/{num_tests} predictions ({sym_time:.1f}s)")
        
        # Phase 2: Program synthesis
        t0 = time.time()
        synth_preds = self.solve_synthesis(task)
        synth_time = time.time() - t0
        
        solved_by_synth = sum(1 for p in synth_preds if p is not None)
        if self.verbose and solved_by_synth:
            print(f"  Synthesis: {solved_by_synth}/{num_tests} predictions ({synth_time:.1f}s)")
        
        # Phase 3: ConvNet TTT — always run (symbolic/synth are unreliable on eval)
        t0 = time.time()
        convnet_preds, convnet_confs = self.solve_convnet(task, num_aug, num_steps, num_vote)
        conv_time = time.time() - t0
        if self.verbose:
            print(f"  ConvNet: {conv_time:.1f}s")
        
        # Phase 4: Post-process ConvNet predictions using output constraints
        postprocessed_preds = [None] * num_tests
        try:
            from arc_postprocess import postprocess_prediction, learn_output_constraints
            pairs = [(np.array(ex['input']), np.array(ex['output'])) for ex in task['train']]
            constraints = learn_output_constraints(pairs)
            
            for i in range(num_tests):
                if convnet_preds[i] is not None and convnet_confs[i] is not None:
                    test_inp = np.array(task["test"][i]["input"], dtype=np.uint8)
                    pp_candidates = postprocess_prediction(
                        convnet_preds[i], convnet_confs[i], pairs, test_inp,
                        verbose=self.verbose
                    )
                    if len(pp_candidates) > 1:
                        # First candidate is the best post-processed version
                        postprocessed_preds[i] = pp_candidates[1]  # Skip [0] which is original
        except Exception as e:
            if self.verbose:
                print(f"  Post-process error: {e}")
        
        # Combine guesses (2 per test input)
        result = []
        for i in range(num_tests):
            candidates = []
            
            # Priority: pattern > symbolic > synthesis > convnet-postprocessed > convnet
            if pattern_preds[i] is not None:
                candidates.append(pattern_preds[i])
            if symbolic_preds[i] is not None:
                candidates.append(symbolic_preds[i])
            if synth_preds[i] is not None:
                candidates.append(synth_preds[i])
            if postprocessed_preds[i] is not None:
                candidates.append(postprocessed_preds[i])
            if convnet_preds[i] is not None:
                candidates.append(convnet_preds[i])
            
            # Fallback: return input
            test_inp = np.array(task["test"][i]["input"], dtype=np.uint8)
            
            if len(candidates) == 0:
                result.append((test_inp, test_inp))
            elif len(candidates) == 1:
                result.append((candidates[0], test_inp))
            else:
                # Use first two distinct predictions
                g1 = candidates[0]
                g2 = None
                for c in candidates[1:]:
                    if c.shape != g1.shape or not np.array_equal(c, g1):
                        g2 = c
                        break
                result.append((g1, g2 if g2 is not None else test_inp))
        
        return result


def evaluate(
    data_dir: str,
    device: torch.device,
    max_tasks: Optional[int] = None,
    conv_aug: int = 100,
    conv_steps: int = 1000,
    conv_vote: int = 32,
    verbose: bool = True,
) -> Dict:
    """Run combined evaluation."""
    tasks = load_tasks(data_dir)
    task_items = list(tasks.items())[:max_tasks] if max_tasks else list(tasks.items())
    
    solver = CombinedSolver(device, verbose=verbose)
    
    print(f"Combined evaluation: {len(task_items)} tasks")
    print(f"ConvNet params: aug={conv_aug}, steps={conv_steps}, vote={conv_vote}")
    print("="*60)
    
    results = {}
    correct_g1 = 0  # Correct with guess 1
    correct_g2 = 0  # Correct with guess 2 (but not guess 1)
    total = 0
    
    for i, (task_id, task) in enumerate(task_items):
        t0 = time.time()
        
        try:
            guesses = solver.solve(task_id, task, num_aug=conv_aug, num_steps=conv_steps, num_vote=conv_vote)
            
            task_correct = True
            for j, ((g1, g2), test_pair) in enumerate(zip(guesses, task["test"])):
                expected = np.array(test_pair.get("output", [[0]]), dtype=np.uint8) if "output" in test_pair else None
                total += 1
                
                if expected is not None:
                    g1_match = g1.shape == expected.shape and np.array_equal(g1, expected)
                    g2_match = g2 is not None and g2.shape == expected.shape and np.array_equal(g2, expected)
                    
                    if g1_match:
                        correct_g1 += 1
                    elif g2_match:
                        correct_g2 += 1
                    else:
                        task_correct = False
                        if verbose:
                            if g1.shape == expected.shape:
                                cell_match = (g1 == expected).sum()
                                print(f"    Test {j} g1: {cell_match}/{expected.size} cells ({cell_match/expected.size:.1%})")
                            else:
                                print(f"    Test {j} g1: shape {g1.shape} vs {expected.shape}")
                else:
                    task_correct = False
            
            elapsed = time.time() - t0
            status = "✓" if task_correct else "✗"
            score = correct_g1 + correct_g2
            print(f"[{i+1}/{len(task_items)}] {task_id}: {status} ({elapsed:.1f}s) [{score}/{total}]")
            
            results[task_id] = {
                "guesses": [
                    {"attempt_1": g1.tolist(), "attempt_2": g2.tolist() if g2 is not None else g1.tolist()}
                    for g1, g2 in guesses
                ],
                "correct": task_correct,
                "time": elapsed,
            }
            
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(task_items)}] {task_id}: ERROR ({e}) ({elapsed:.1f}s)")
            results[task_id] = {"error": str(e), "correct": False, "time": elapsed}
            total += len(task["test"])
    
    total_correct = correct_g1 + correct_g2
    print(f"\n{'='*60}")
    print(f"Results: {total_correct}/{total} ({total_correct/max(total,1):.1%})")
    print(f"  Guess 1 correct: {correct_g1}")
    print(f"  Guess 2 correct: {correct_g2}")
    print(f"{'='*60}")
    
    return {
        "correct_g1": correct_g1,
        "correct_g2": correct_g2,
        "total_correct": total_correct,
        "total": total,
        "accuracy": total_correct / max(total, 1),
        "per_task": results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combined ARC solver evaluation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--conv_aug", type=int, default=100)
    parser.add_argument("--conv_steps", type=int, default=1000)
    parser.add_argument("--conv_vote", type=int, default=32)
    parser.add_argument("--output", type=str, default="combined_eval_results.json")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    
    results = evaluate(
        args.data_dir, device,
        max_tasks=args.max_tasks,
        conv_aug=args.conv_aug,
        conv_steps=args.conv_steps,
        conv_vote=args.conv_vote,
        verbose=args.verbose,
    )
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {args.output}")
