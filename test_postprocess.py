"""
Test ConvNet + post-processor on best near-miss eval tasks.
Reverted change-weighted loss; added constraint-based post-processing.
"""
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from arc_conv_ttt import solve_task
from arc_postprocess import (
    postprocess_prediction, learn_output_constraints,
    apply_symmetry_fix, apply_color_count_fix
)


def test_near_misses():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation")
    
    # Tasks with best ConvNet accuracy (from prior run)
    targets = [
        "88e364bc",  # 98.5% (6 wrong on 20x20) — has same_color_counts_as_input
        "409aa875",  # 96.5% (9 wrong on 16x16)
        "dd6b8c4b",  # 93.4%/82.6% (8/21 wrong on 11x11)
        "8e5c0c38",  # 96.7%/97.7% (16/11 wrong on 22x22)
        "e376de54",  # 94.5% (14 wrong on 16x16) — was 98.8% before weighted loss
        "135a2760",  # 97.0% (25 wrong on 22x22)
        "9bbf930d",  # 97.3% (13 wrong on 19x19)
        "332f06d7",  # 96.0% (16 wrong on 16x16)
    ]
    
    total_solved = 0
    total_tests = 0
    
    for task_id in targets:
        fpath = os.path.join(data_dir, f"{task_id}.json")
        if not os.path.exists(fpath):
            print(f"  {task_id}: file not found, skipping")
            continue
        
        with open(fpath) as f:
            task = json.load(f)
        
        pairs = [(np.array(ex['input']), np.array(ex['output'])) for ex in task['train']]
        constraints = learn_output_constraints(pairs)
        c_names = [k for k in constraints if not k.startswith('_')]
        print(f"\n{'='*60}")
        print(f"Task {task_id} — constraints: {c_names}")
        
        t0 = time.time()
        predictions, confidences = solve_task(
            task, device,
            num_aug=150,
            num_steps=1500,
            num_vote=48,
            verbose=True,
        )
        conv_time = time.time() - t0
        
        for ti, (pred, conf) in enumerate(zip(predictions, confidences)):
            test_pair = task["test"][ti]
            test_inp = np.array(test_pair["input"], dtype=np.uint8)
            expected = np.array(test_pair["output"], dtype=np.uint8) if "output" in test_pair else None
            total_tests += 1
            
            # Check raw ConvNet accuracy
            if expected is not None and pred.shape == expected.shape:
                raw_wrong = (pred != expected).sum()
                raw_total = expected.size
                print(f"  {task_id} test{ti}: ConvNet raw = {(raw_total-raw_wrong)/raw_total:.1%} ({raw_wrong} wrong)")
            
            # Run post-processor
            pp_candidates = postprocess_prediction(
                pred, conf, pairs, test_inp, verbose=True
            )
            
            # Check all candidates
            best_wrong = raw_wrong if expected is not None else float('inf')
            best_idx = 0
            solved = False
            
            for ci, cand in enumerate(pp_candidates):
                if expected is not None and cand.shape == expected.shape:
                    wrong = (cand != expected).sum()
                    if wrong == 0:
                        print(f"  ✅ SOLVED by candidate {ci}!")
                        solved = True
                        total_solved += 1
                        break
                    if wrong < best_wrong:
                        best_wrong = wrong
                        best_idx = ci
            
            if not solved and expected is not None:
                improvement = raw_wrong - best_wrong
                print(f"  Best post-processed: candidate {best_idx}, {best_wrong} wrong "
                      f"(improvement: {improvement} cells)")
                
                # Show what the ConvNet gets wrong
                if best_wrong <= 15:
                    wrong_mask = pred != expected
                    wrong_positions = list(zip(*np.where(wrong_mask)))
                    for r, c in wrong_positions[:10]:
                        print(f"    ({r},{c}): predicted={pred[r,c]} expected={expected[r,c]}")
        
        print(f"  Time: {conv_time:.1f}s")
    
    print(f"\n{'='*60}")
    print(f"Total: {total_solved}/{total_tests} exact matches")


if __name__ == "__main__":
    test_near_misses()
