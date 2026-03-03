#!/usr/bin/env python3
"""Brute-force search for ARC exact matches using confidence-based cell search.

For each task:
1. Run model inference to get per-cell softmax probabilities
2. Identify the N lowest-confidence cells
3. Try top-K color alternatives for those cells
4. Check if any combination produces an exact match

This can find exact matches when the model is "close but not quite" —
i.e., when the correct answer is in the top-K predictions for wrong cells.
"""
import argparse
import json
import glob
import logging
import os
import time
from itertools import product
from typing import Dict, List, Tuple

import torch
import tiktoken

from config import get_config
from model import OctoTetrahedralModel
from grid_head import GridPredictionHead, MAX_GRID_SIZE
from eval_arc_grid import encode_context, grid_to_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device):
    """Load model and grid head from checkpoint."""
    cfg = get_config()
    model = OctoTetrahedralModel(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # Detect num_layers from checkpoint
    layer_keys = [k for k in ckpt['grid_head_state_dict'] if 'decoder.layers' in k]
    max_layer = max(int(k.split('.')[2]) for k in layer_keys) + 1
    grid_head = GridPredictionHead(hidden_dim=cfg.model.hidden_dim, num_layers=max_layer)
    grid_head.load_state_dict(ckpt['grid_head_state_dict'])

    model.to(device).eval()
    grid_head.to(device).eval()
    return model, grid_head, cfg


def predict_with_confidence(
    model, grid_head, task: Dict, tokenizer,
    max_seq_len: int, device: torch.device
) -> List[Dict]:
    """Get predictions with per-cell confidence for each test example."""
    train_examples = task['train']

    # Collect dimension candidates from training outputs + inputs
    dim_candidates = set()
    for ex in train_examples:
        out = ex['output']
        dim_candidates.add((len(out), len(out[0]) if out else 0))
        inp = ex['input']
        dim_candidates.add((len(inp), len(inp[0]) if inp else 0))

    results = []
    for test_ex in task['test']:
        test_input = test_ex['input']
        inp_h, inp_w = len(test_input), len(test_input[0]) if test_input else 0
        dim_candidates.add((inp_h, inp_w))

        input_ids, attn_mask = encode_context(
            train_examples, test_input, tokenizer, max_seq_len)
        input_ids = input_ids.unsqueeze(0).to(device)
        attn_mask = attn_mask.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = output['hidden_states']
            pred = grid_head(hidden)

        grid_logits = pred['grid_logits'][0].cpu()  # [30, 30, 10]
        dim_logits = pred['dim_logits'][0].cpu()     # [2, 30]

        # Model's predicted dims
        pred_h = dim_logits[0].argmax().item() + 1
        pred_w = dim_logits[1].argmax().item() + 1
        all_dims = [(pred_h, pred_w)] + list(dim_candidates)

        # For each dimension candidate, compute prediction and confidence
        best_result = None
        best_avg_conf = -1

        for cand_h, cand_w in all_dims:
            if cand_h <= 0 or cand_w <= 0 or cand_h > 30 or cand_w > 30:
                continue
            probs = torch.softmax(grid_logits[:cand_h, :cand_w, :], dim=-1)
            pred_grid = probs.argmax(dim=-1)  # [h, w]
            confidences = probs.max(dim=-1).values  # [h, w]
            avg_conf = confidences.mean().item()

            if avg_conf > best_avg_conf:
                best_avg_conf = avg_conf
                best_result = {
                    'pred_grid': pred_grid,
                    'probs': probs,
                    'confidences': confidences,
                    'dims': (cand_h, cand_w),
                    'avg_conf': avg_conf,
                }

        # Also include target-dim result if we know it
        if 'output' in test_ex:
            target = test_ex['output']
            th, tw = len(target), len(target[0])
            probs = torch.softmax(grid_logits[:th, :tw, :], dim=-1)
            results.append({
                'pred_grid': probs.argmax(dim=-1),
                'probs': probs,
                'confidences': probs.max(dim=-1).values,
                'dims': (th, tw),
                'target': torch.tensor(target),
                'best_conf_result': best_result,
                'grid_logits': grid_logits,
            })
        else:
            best_result['target'] = None
            best_result['grid_logits'] = grid_logits
            results.append(best_result)

    return results


def bruteforce_search(
    probs: torch.Tensor,
    pred_grid: torch.Tensor,
    target: torch.Tensor,
    max_search_cells: int = 12,
    top_k: int = 3,
) -> Tuple[bool, torch.Tensor, int]:
    """Try top-K colors for lowest-confidence cells to find exact match.

    Returns (found_exact, best_grid, combinations_tried).
    """
    h, w = target.shape
    wrong_mask = pred_grid != target
    n_wrong = wrong_mask.sum().item()

    if n_wrong == 0:
        return True, pred_grid.clone(), 0

    # Get all cells sorted by confidence (ascending = least confident first)
    confs = probs.max(dim=-1).values  # [h, w]
    flat_confs = confs.reshape(-1)
    sorted_indices = flat_confs.argsort()  # least confident first

    # Select cells to search: prioritize wrong cells, then low-confidence
    search_cells = []
    wrong_positions = set()
    for r in range(h):
        for c in range(w):
            if wrong_mask[r, c]:
                wrong_positions.add((r, c))

    # Add wrong cells first (sorted by confidence)
    for idx in sorted_indices:
        r, c = idx.item() // w, idx.item() % w
        if (r, c) in wrong_positions and len(search_cells) < max_search_cells:
            top_colors = torch.topk(probs[r, c], min(top_k, 10)).indices.tolist()
            search_cells.append((r, c, top_colors))

    # Add low-confidence correct cells (they might need flipping too)
    for idx in sorted_indices:
        r, c = idx.item() // w, idx.item() % w
        if (r, c) not in wrong_positions and len(search_cells) < max_search_cells:
            top_colors = torch.topk(probs[r, c], min(top_k, 10)).indices.tolist()
            search_cells.append((r, c, top_colors))

    if not search_cells:
        return False, pred_grid.clone(), 0

    # Limit search space
    n_cells = len(search_cells)
    total_combos = 1
    for _, _, colors in search_cells:
        total_combos *= len(colors)

    if total_combos > 1_000_000:
        # Too many combos — reduce cells
        while total_combos > 1_000_000 and n_cells > 1:
            n_cells -= 1
            total_combos = 1
            for _, _, colors in search_cells[:n_cells]:
                total_combos *= len(colors)
        search_cells = search_cells[:n_cells]

    # Generate all combinations
    color_options = [colors for _, _, colors in search_cells]
    best_grid = pred_grid.clone()
    combos_tried = 0

    for combo in product(*color_options):
        test_grid = pred_grid.clone()
        for i, (r, c, _) in enumerate(search_cells):
            test_grid[r, c] = combo[i]
        combos_tried += 1

        if torch.equal(test_grid, target):
            return True, test_grid, combos_tried

    return False, best_grid, combos_tried


def main():
    parser = argparse.ArgumentParser(description='Brute-force ARC grid search')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/grid_run3/grid_best.pt')
    parser.add_argument('--split', type=str, default='evaluation')
    parser.add_argument('--max-tasks', type=int, default=120)
    parser.add_argument('--max-search-cells', type=int, default=10,
                        help='Max cells to search (combos = top_k^N)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Top-K colors to try per cell')
    parser.add_argument('--min-cell-acc', type=float, default=0.7,
                        help='Only brute-force tasks with >= this cell accuracy')
    parser.add_argument('--output', type=str, default='grid_eval_bruteforce.json')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, grid_head, cfg = load_model(args.checkpoint, device)
    tokenizer = tiktoken.get_encoding('cl100k_base')

    # Find ARC data
    arc_dirs = [
        f'ARC_AMD_TRANSFER/data/ARC-AGI-2/data/{args.split}',
        f'ARC_AMD_TRANSFER/data/ARC-AGI/data/{args.split}',
    ]
    task_files = []
    for d in arc_dirs:
        task_files.extend(sorted(glob.glob(os.path.join(d, '*.json'))))
    task_files = task_files[:args.max_tasks]
    logger.info(f"Evaluating {len(task_files)} tasks on {device}")

    total_solved = 0
    total_bf_solved = 0
    total_examples = 0
    task_results = {}
    t_start = time.time()

    for i, tf in enumerate(task_files):
        task_id = os.path.basename(tf).replace('.json', '')
        with open(tf) as f:
            task = json.load(f)

        preds = predict_with_confidence(
            model, grid_head, task, tokenizer, 2048, device)

        task_exact = 0
        task_bf_exact = 0
        task_details = []

        for j, p in enumerate(preds):
            total_examples += 1
            target = p['target']
            if target is None:
                continue

            pred_grid = p['pred_grid']
            h, w = target.shape
            wrong = (pred_grid[:h, :w] != target).sum().item()
            total = h * w
            cell_acc = 1.0 - wrong / total

            # Direct exact match?
            direct_exact = torch.equal(pred_grid[:h, :w], target)
            if direct_exact:
                task_exact += 1
                task_bf_exact += 1
                task_details.append({
                    'test_idx': j, 'cell_acc': cell_acc,
                    'wrong_cells': 0, 'direct_exact': True,
                    'bf_exact': True, 'combos_tried': 0,
                })
                continue

            # Brute-force search if cell_acc >= threshold
            bf_exact = False
            combos = 0
            if cell_acc >= args.min_cell_acc:
                probs = p['probs'][:h, :w]
                bf_exact, bf_grid, combos = bruteforce_search(
                    probs, pred_grid[:h, :w], target,
                    max_search_cells=args.max_search_cells,
                    top_k=args.top_k)
                if bf_exact:
                    task_bf_exact += 1

            task_details.append({
                'test_idx': j, 'cell_acc': cell_acc,
                'wrong_cells': wrong, 'direct_exact': False,
                'bf_exact': bf_exact, 'combos_tried': combos,
                'dims': [h, w],
            })

        total_solved += (1 if task_exact == len(preds) else 0)
        total_bf_solved += (1 if task_bf_exact == len(preds) else 0)

        bf_marker = "✅ BF-SOLVED" if task_bf_exact == len(preds) else ""
        details_str = " | ".join(
            f"{'✓' if d['bf_exact'] else '✗'} {d['cell_acc']:.1%} ({d['wrong_cells']}w, {d['combos_tried']}c)"
            for d in task_details
        )
        logger.info(f"[{i+1}/{len(task_files)}] {task_id}: {details_str} {bf_marker}")

        task_results[task_id] = {
            'direct_solved': task_exact == len(preds),
            'bf_solved': task_bf_exact == len(preds),
            'details': task_details,
        }

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Brute-Force ARC Evaluation ({args.split})")
    logger.info(f"  Tasks: {len(task_files)}")
    logger.info(f"  Direct exact match: {total_solved}/{len(task_files)}")
    logger.info(f"  With brute-force:   {total_bf_solved}/{len(task_files)}")
    logger.info(f"  Search params: top_k={args.top_k}, max_cells={args.max_search_cells}")
    logger.info(f"  Min cell acc threshold: {args.min_cell_acc:.0%}")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/len(task_files):.1f}s/task)")
    logger.info("=" * 60)

    output = {
        'split': args.split,
        'checkpoint': args.checkpoint,
        'direct_solved': total_solved,
        'bf_solved': total_bf_solved,
        'n_tasks': len(task_files),
        'top_k': args.top_k,
        'max_search_cells': args.max_search_cells,
        'min_cell_acc': args.min_cell_acc,
        'elapsed': elapsed,
        'tasks': task_results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
