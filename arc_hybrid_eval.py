#!/usr/bin/env python3
"""
Hybrid ARC Solver — combines program synthesis + neural model + brute-force search.

Strategy:
1. Try program synthesis (exact rule induction)
2. If no rule found, use neural model prediction
3. Apply brute-force search to neural prediction for near-misses
4. Return best result
"""

import json
import glob
import os
import sys
import time
import numpy as np
from typing import List, Tuple, Dict, Optional

# Import synthesizer
from arc_synth import ARCSynthesizer, grid_eq, to_np, to_grid, Grid


def load_neural_model():
    """Load the OctoTetrahedral model + grid head for neural predictions."""
    try:
        import torch
        sys.path.insert(0, '.')
        from model import OctoTetrahedralModel
        from grid_head import GridPredictionHead
        from config import Config

        config = Config()
        device = 'cpu'
        model = OctoTetrahedralModel(config, use_geometric_physics=False).to(device)
        grid_head = GridPredictionHead(hidden_dim=256, max_grid=30, num_layers=3).to(device)

        ckpt_path = 'checkpoints/grid_run3/grid_best.pt'
        if not os.path.exists(ckpt_path):
            print("WARNING: No neural checkpoint found")
            return None, None, None

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        grid_head.load_state_dict(checkpoint['grid_head_state_dict'], strict=False)

        model.eval()
        grid_head.eval()
        return model, grid_head, device

    except Exception as e:
        print(f"WARNING: Could not load neural model: {e}")
        return None, None, None


def neural_predict(model, grid_head, device, task: dict) -> Tuple[Optional[Grid], Optional[np.ndarray]]:
    """Get neural model prediction for a task."""
    import torch
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    # Format task as text
    text_parts = []
    for i, ex in enumerate(task['train']):
        text_parts.append(f"Input {i}: {json.dumps(ex['input'])}")
        text_parts.append(f"Output {i}: {json.dumps(ex['output'])}")
    text_parts.append(f"Test Input: {json.dumps(task['test'][0]['input'])}")
    text = "\n".join(text_parts)

    tokens = enc.encode(text)[:2048]
    input_ids = torch.tensor([tokens], device=device)

    # Get training output dims as candidates
    dim_candidates = set()
    for ex in task['train']:
        out = np.array(ex['output'])
        dim_candidates.add((out.shape[0], out.shape[1]))
    test_inp = np.array(task['test'][0]['input'])
    dim_candidates.add((test_inp.shape[0], test_inp.shape[1]))

    best_pred = None
    best_conf = -1
    best_probs = None

    with torch.no_grad():
        outputs = model(input_ids)
        hidden = outputs['hidden_states']
        result = grid_head(hidden)
        grid_logits = result['grid_logits'][0].cpu()  # [30, 30, 10]
        dim_logits = result['dim_logits'][0].cpu()     # [2, 30]

        # Model's predicted dims
        pred_h = dim_logits[0].argmax().item() + 1
        pred_w = dim_logits[1].argmax().item() + 1
        all_dims = [(pred_h, pred_w)] + list(dim_candidates)

        for h_cand, w_cand in all_dims:
            if h_cand <= 0 or w_cand <= 0 or h_cand > 30 or w_cand > 30:
                continue

            probs = torch.softmax(grid_logits[:h_cand, :w_cand, :], dim=-1)
            pred = probs.argmax(dim=-1).numpy()  # [h, w]
            confidence = probs.max(dim=-1)[0].mean().item()

            if confidence > best_conf:
                best_conf = confidence
                best_pred = pred
                best_probs = probs.numpy()  # [h, w, 10]

    if best_pred is not None:
        return to_grid(best_pred), best_probs
    return None, None


def brute_force_search(pred: np.ndarray, probs: np.ndarray, target: Optional[np.ndarray] = None,
                       top_k: int = 3, max_cells: int = 10, max_combos: int = 1_000_000) -> List[np.ndarray]:
    """Try top-K colors for lowest-confidence cells."""
    h, w = pred.shape
    confidences = probs.max(axis=-1)  # [h, w]

    # Get sorted cell indices by confidence (ascending)
    cell_confs = []
    for r in range(h):
        for c in range(w):
            cell_confs.append((confidences[r, c], r, c))
    cell_confs.sort()

    # Take worst N cells
    n_cells = min(max_cells, len(cell_confs))
    while top_k ** n_cells > max_combos and n_cells > 0:
        n_cells -= 1

    if n_cells == 0:
        return [pred]

    worst_cells = cell_confs[:n_cells]

    # Get top-K colors for each cell
    cell_options = []
    for conf, r, c in worst_cells:
        top_colors = np.argsort(-probs[r, c])[:top_k]
        cell_options.append((r, c, top_colors))

    # Enumerate combinations
    from itertools import product as iter_product
    candidates = []
    for combo in iter_product(*[opts for _, _, opts in cell_options]):
        candidate = pred.copy()
        for idx, (r, c, _) in enumerate(cell_options):
            candidate[r, c] = combo[idx]
        candidates.append(candidate)
        if len(candidates) >= max_combos:
            break

    return candidates


class HybridArcSolver:
    """Combines synthesis + neural + brute-force."""

    def __init__(self, use_neural: bool = True, verbose: bool = False):
        self.synth = ARCSynthesizer(max_time=15.0, verbose=verbose)
        self.verbose = verbose
        self.use_neural = use_neural
        self.model = None
        self.grid_head = None
        self.device = None

        if use_neural:
            self.model, self.grid_head, self.device = load_neural_model()
            if self.model is None:
                print("Neural model not available, using synthesis only")
                self.use_neural = False

    def solve(self, task: dict) -> List[Grid]:
        """Solve an ARC task using hybrid approach."""
        # Phase 1: Program synthesis
        try:
            preds = self.synth.synthesize(task)
            # Check if synthesis found a real rule (not just returning input)
            test_inp = task['test'][0]['input']
            if not grid_eq(preds[0], test_inp):
                if self.verbose:
                    print("  → Synthesis found a rule")
                return preds
        except Exception:
            pass

        # Phase 2: Neural prediction + brute-force
        if self.use_neural:
            try:
                pred, probs = neural_predict(self.model, self.grid_head, self.device, task)
                if pred is not None:
                    if self.verbose:
                        print("  → Using neural prediction")
                    return [pred]
            except Exception as e:
                if self.verbose:
                    print(f"  → Neural failed: {e}")

        # Fallback: return input
        return [task['test'][j]['input'] for j in range(len(task['test']))]


def evaluate(data_dir: str, max_tasks: int = 0, use_neural: bool = True, verbose: bool = False):
    """Evaluate hybrid solver."""
    task_files = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    if max_tasks > 0:
        task_files = task_files[:max_tasks]

    solver = HybridArcSolver(use_neural=use_neural, verbose=verbose)
    results = {}
    synth_solved = 0
    neural_solved = 0

    for i, tf in enumerate(task_files):
        task_id = os.path.basename(tf).replace('.json', '')
        with open(tf) as f:
            task = json.load(f)

        preds = solver.solve(task)

        task_solved = True
        for j, test_ex in enumerate(task['test']):
            if 'output' in test_ex:
                if j >= len(preds) or not grid_eq(preds[j], test_ex['output']):
                    task_solved = False

        results[task_id] = task_solved

        if (i + 1) % 30 == 0 or i == len(task_files) - 1:
            solved = sum(results.values())
            print(f"  [{i+1}/{len(task_files)}] solved={solved}/{i+1} ({100*solved/(i+1):.1f}%)")

    total = sum(results.values())
    n = len(task_files)
    print(f"\n{'='*50}")
    print(f"Hybrid Solver Results: {total}/{n} ({100*total/n:.1f}%)")
    print(f"{'='*50}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation')
    parser.add_argument('--max-tasks', type=int, default=120)
    parser.add_argument('--no-neural', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    results = evaluate(args.data, max_tasks=args.max_tasks,
                      use_neural=not args.no_neural, verbose=args.verbose)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")
