#!/usr/bin/env python3
"""
Multi-seed ensemble ConvNet for ARC-AGI-2.

Strategy: Train K models with different random seeds, take per-cell majority vote.
If errors are partially random across seeds, voting can correct them.
"""
import os
import sys
import json
import time
import numpy as np
import torch
from typing import List, Tuple, Optional
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from arc_conv_ttt import (
    solve_task, train_same_size, predict_same_size, analyze_task,
    NUM_COLORS
)
from arc_postprocess import postprocess_prediction, learn_output_constraints


def majority_vote(predictions: List[np.ndarray]) -> np.ndarray:
    """Per-cell majority vote across multiple predictions."""
    if len(predictions) == 1:
        return predictions[0]
    
    h, w = predictions[0].shape
    result = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            votes = [p[i, j] for p in predictions if p.shape == (h, w)]
            if votes:
                result[i, j] = Counter(votes).most_common(1)[0][0]
    
    return result


def weighted_vote(predictions: List[np.ndarray], 
                  confidences: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell weighted vote using logit sums."""
    h, w = predictions[0].shape
    combined_logits = np.zeros((NUM_COLORS, h, w), dtype=np.float64)
    
    for conf in confidences:
        if conf.shape[1:] == (h, w):
            combined_logits += conf
    
    result = combined_logits.argmax(axis=0).astype(np.uint8)
    return result, combined_logits


def solve_ensemble(
    task: dict,
    device: torch.device,
    num_seeds: int = 5,
    num_aug: int = 150,
    num_steps: int = 1500,
    num_vote: int = 48,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Solve task with multi-seed ensemble."""
    num_tests = len(task['test'])
    
    all_preds = [[] for _ in range(num_tests)]
    all_confs = [[] for _ in range(num_tests)]
    
    for seed_idx in range(num_seeds):
        if verbose:
            print(f"  Seed {seed_idx+1}/{num_seeds}")
        
        # Set different seed for each run
        torch.manual_seed(42 + seed_idx * 17)
        np.random.seed(42 + seed_idx * 17)
        
        try:
            preds, confs = solve_task(
                task, device,
                num_aug=num_aug,
                num_steps=num_steps,
                num_vote=num_vote,
                verbose=verbose,
            )
            
            for ti in range(num_tests):
                if ti < len(preds):
                    all_preds[ti].append(preds[ti])
                    all_confs[ti].append(confs[ti])
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
    
    # Combine via weighted vote (sum logits across seeds)
    results = []
    result_confs = []
    for ti in range(num_tests):
        if all_preds[ti]:
            # Method 1: Weighted vote (sum logits)
            w_pred, w_conf = weighted_vote(all_preds[ti], all_confs[ti])
            results.append(w_pred)
            result_confs.append(w_conf)
            
            if verbose:
                # Also compute majority vote for comparison
                m_pred = majority_vote(all_preds[ti])
                diff_wm = (w_pred != m_pred).sum()
                
                # Show per-seed agreement
                h, w = all_preds[ti][0].shape
                agreement = np.zeros((h, w), dtype=int)
                for p in all_preds[ti]:
                    agreement += (p == w_pred)
                min_agree = agreement.min()
                mean_agree = agreement.mean()
                print(f"    Test {ti}: min_agreement={min_agree}/{len(all_preds[ti])}, "
                      f"mean={mean_agree:.1f}, weighted≠majority={diff_wm} cells")
        else:
            inp = np.array(task['test'][ti]['input'], dtype=np.uint8)
            results.append(inp)
            result_confs.append(np.zeros((NUM_COLORS, *inp.shape)))
    
    return results, result_confs


def test_ensemble():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation")
    
    targets = [
        "e376de54",  # 4 wrong (plain ConvNet)
        "88e364bc",  # 6 wrong
        "dd6b8c4b",  # 7 wrong
    ]
    
    total_solved = 0
    total_tests = 0
    
    for task_id in targets:
        fpath = os.path.join(data_dir, f"{task_id}.json")
        if not os.path.exists(fpath):
            continue
        
        with open(fpath) as f:
            task = json.load(f)
        
        pairs = [(np.array(ex['input']), np.array(ex['output'])) for ex in task['train']]
        
        print(f"\n{'='*60}")
        print(f"Task {task_id}")
        
        t0 = time.time()
        predictions, confidences = solve_ensemble(
            task, device,
            num_seeds=5,
            num_aug=150,
            num_steps=1500,
            num_vote=48,
            verbose=True,
        )
        elapsed = time.time() - t0
        
        for ti, (pred, conf) in enumerate(zip(predictions, confidences)):
            test_pair = task['test'][ti]
            total_tests += 1
            
            if 'output' in test_pair:
                expected = np.array(test_pair['output'], dtype=np.uint8)
                if pred.shape == expected.shape:
                    wrong = (pred != expected).sum()
                    total = expected.size
                    if wrong == 0:
                        print(f"  ✅ test{ti}: SOLVED!")
                        total_solved += 1
                    else:
                        print(f"  test{ti}: {wrong} wrong ({(total-wrong)/total:.1%})")
                        if wrong <= 10:
                            for r, c in zip(*np.where(pred != expected)):
                                print(f"    ({r},{c}): pred={pred[r,c]} exp={expected[r,c]}")
                        
                        # Also try post-processing
                        pp_cands = postprocess_prediction(pred, conf, pairs,
                                                          np.array(test_pair['input'], dtype=np.uint8))
                        best_pp = wrong
                        for ci, cand in enumerate(pp_cands):
                            if cand.shape == expected.shape:
                                pp_wrong = (cand != expected).sum()
                                if pp_wrong == 0:
                                    print(f"  ✅ test{ti}: SOLVED by post-processing candidate {ci}!")
                                    total_solved += 1
                                    break
                                best_pp = min(best_pp, pp_wrong)
                        if best_pp < wrong:
                            print(f"    Post-processing improved: {wrong} → {best_pp}")
        
        print(f"  Time: {elapsed:.1f}s")
    
    print(f"\n{'='*60}")
    print(f"Total: {total_solved}/{total_tests} exact matches")


if __name__ == "__main__":
    test_ensemble()
