#!/usr/bin/env python3
"""
Fast heuristic ARC solvers — targets patterns the DSL misses.
Each solver is a simple Python function: task → prediction or None.
"""

import json
import os
import sys
import time
import numpy as np
from typing import Optional, List, Tuple, Dict
from itertools import product


def grids_match(pred, target) -> bool:
    if pred is None or target is None:
        return False
    a = np.array(pred)
    b = np.array(target)
    return a.shape == b.shape and np.array_equal(a, b)


# ═══════════════════════════════════════════════════════════════
# Solver 1: Sub-grid extraction (for smaller output)
# ═══════════════════════════════════════════════════════════════

def solve_subgrid_extract(task: dict) -> Optional[list]:
    """For tasks where output is a sub-region of input."""
    train = task['train']
    test_inp = np.array(task['test'][0]['input'])
    
    out_shape = np.array(train[0]['output']).shape
    
    # All training outputs must be the same shape
    for ex in train:
        if np.array(ex['output']).shape != out_shape:
            return None
    
    oh, ow = out_shape
    
    # Strategy: find a rule for extracting the subgrid
    # Try fixed position
    for r in range(test_inp.shape[0] - oh + 1):
        for c in range(test_inp.shape[1] - ow + 1):
            # Check if this position works for ALL training examples
            all_match = True
            for ex in train:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                if r + oh > inp.shape[0] or c + ow > inp.shape[1]:
                    all_match = False
                    break
                if not np.array_equal(inp[r:r+oh, c:c+ow], out):
                    all_match = False
                    break
            if all_match:
                return test_inp[r:r+oh, c:c+ow].tolist()
    
    return None


# ═══════════════════════════════════════════════════════════════
# Solver 2: Unique non-background region extraction
# ═══════════════════════════════════════════════════════════════

def solve_unique_region(task: dict) -> Optional[list]:
    """Extract a unique colored region as the output."""
    train = task['train']
    test_inp = np.array(task['test'][0]['input'])
    
    def find_bbox_of_color(grid, color):
        rows, cols = np.where(grid == color)
        if len(rows) == 0:
            return None
        return rows.min(), cols.min(), rows.max() + 1, cols.max() + 1
    
    # Find which strategy works for training
    # Try: extract bbox of the least common non-zero color
    for ex in train:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        oh, ow = out.shape
        
        colors, counts = np.unique(inp, return_counts=True)
        # Sort by count (ascending) — rarest first
        sorted_colors = colors[np.argsort(counts)]
        
        for color in sorted_colors:
            if color == 0:
                continue
            bbox = find_bbox_of_color(inp, color)
            if bbox is None:
                continue
            r1, c1, r2, c2 = bbox
            sub = inp[r1:r2, c1:c2]
            if sub.shape == out.shape and np.array_equal(sub, out):
                # This color's bbox works! Check all training examples
                strategy_color_rank = list(sorted_colors).index(color)
                break
        else:
            return None
    
    return None  # Too complex for simple version


# ═══════════════════════════════════════════════════════════════
# Solver 3: Tiling (for larger output)
# ═══════════════════════════════════════════════════════════════

def solve_tiling(task: dict) -> Optional[list]:
    """Output is input tiled/repeated N times."""
    train = task['train']
    
    for ex in train:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        ih, iw = inp.shape
        oh, ow = out.shape
        
        if oh < ih or ow < iw:
            return None
        if oh % ih != 0 or ow % iw != 0:
            return None
    
    # Determine tile ratios
    inp0 = np.array(train[0]['input'])
    out0 = np.array(train[0]['output'])
    rh, rw = out0.shape[0] // inp0.shape[0], out0.shape[1] // inp0.shape[1]
    
    # Verify consistent ratio
    for ex in train:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        if out.shape[0] // inp.shape[0] != rh or out.shape[1] // inp.shape[1] != rw:
            return None
    
    # Check if it's simple tiling
    for ex in train:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        tiled = np.tile(inp, (rh, rw))
        if not np.array_equal(tiled, out):
            return None
    
    test_inp = np.array(task['test'][0]['input'])
    return np.tile(test_inp, (rh, rw)).tolist()


# ═══════════════════════════════════════════════════════════════
# Solver 4: Majority-pixel / overlay detection  
# ═══════════════════════════════════════════════════════════════

def solve_majority_rule(task: dict) -> Optional[list]:
    """Tasks where output = input with certain cells changed by a rule."""
    train = task['train']
    
    # Must be same-size
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    # Find cells that change
    # Strategy: see if changing cells follow a pattern based on their value/neighborhood
    
    # Simple: does the background (0) stay 0 and non-zero stays same?
    bg_stays = True
    for ex in train:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        if not np.array_equal(inp, out):
            bg_stays = False
            break
    
    if bg_stays:
        return np.array(task['test'][0]['input']).tolist()
    
    return None


# ═══════════════════════════════════════════════════════════════
# Solver 5: Gravity / flood fill
# ═══════════════════════════════════════════════════════════════

def solve_gravity(task: dict) -> Optional[list]:
    """Drop colored cells down (gravity) or to a side."""
    train = task['train']
    
    # Must be same-size
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    directions = ['down', 'up', 'left', 'right']
    
    for direction in directions:
        all_match = True
        for ex in train:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            pred = apply_gravity(inp, direction)
            if not np.array_equal(pred, out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            return apply_gravity(test_inp, direction).tolist()
    
    return None


def apply_gravity(grid: np.ndarray, direction: str) -> np.ndarray:
    """Apply gravity to non-zero cells in specified direction."""
    result = np.zeros_like(grid)
    h, w = grid.shape
    
    if direction == 'down':
        for c in range(w):
            col = grid[:, c]
            non_zero = col[col != 0]
            result[h - len(non_zero):, c] = non_zero
    elif direction == 'up':
        for c in range(w):
            col = grid[:, c]
            non_zero = col[col != 0]
            result[:len(non_zero), c] = non_zero
    elif direction == 'right':
        for r in range(h):
            row = grid[r, :]
            non_zero = row[row != 0]
            result[r, w - len(non_zero):] = non_zero
    elif direction == 'left':
        for r in range(h):
            row = grid[r, :]
            non_zero = row[row != 0]
            result[r, :len(non_zero)] = non_zero
    
    return result


# ═══════════════════════════════════════════════════════════════
# Solver 6: Row/column sorting
# ═══════════════════════════════════════════════════════════════

def solve_sorting(task: dict) -> Optional[list]:
    """Sort rows or columns by some criterion."""
    train = task['train']
    
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    # Try sorting rows
    for key_fn_name, key_fn in [
        ('count_nonzero', lambda row: np.count_nonzero(row)),
        ('sum', lambda row: np.sum(row)),
        ('max', lambda row: np.max(row)),
    ]:
        for reverse in [False, True]:
            all_match = True
            for ex in train:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                
                rows = list(range(inp.shape[0]))
                rows.sort(key=lambda r: key_fn(inp[r]), reverse=reverse)
                pred = inp[rows]
                if not np.array_equal(pred, out):
                    all_match = False
                    break
            
            if all_match:
                test_inp = np.array(task['test'][0]['input'])
                rows = list(range(test_inp.shape[0]))
                rows.sort(key=lambda r: key_fn(test_inp[r]), reverse=reverse)
                return test_inp[rows].tolist()
    
    # Try sorting columns
    for key_fn_name, key_fn in [
        ('count_nonzero', lambda col: np.count_nonzero(col)),
        ('sum', lambda col: np.sum(col)),
        ('max', lambda col: np.max(col)),
    ]:
        for reverse in [False, True]:
            all_match = True
            for ex in train:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                
                cols = list(range(inp.shape[1]))
                cols.sort(key=lambda c: key_fn(inp[:, c]), reverse=reverse)
                pred = inp[:, cols]
                if not np.array_equal(pred, out):
                    all_match = False
                    break
            
            if all_match:
                test_inp = np.array(task['test'][0]['input'])
                cols = list(range(test_inp.shape[1]))
                cols.sort(key=lambda c: key_fn(test_inp[:, c]), reverse=reverse)
                return test_inp[:, cols].tolist()
    
    return None


# ═══════════════════════════════════════════════════════════════
# Solver 7: Pixel-level spatial correspondence
# ═══════════════════════════════════════════════════════════════

def solve_spatial_correspondence(task: dict) -> Optional[list]:
    """Find a fixed spatial mapping: out[r,c] = in[f(r,c)]."""
    train = task['train']
    
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    h, w = np.array(train[0]['input']).shape
    
    # For each output position, find which input position it maps from
    # Try: out[r,c] = in[r+dr, c+dc] for fixed (dr, dc) — shift
    for dr in range(-h+1, h):
        for dc in range(-w+1, w):
            if dr == 0 and dc == 0:
                continue
            all_match = True
            for ex in train:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                if inp.shape != (h, w) or out.shape != (h, w):
                    all_match = False
                    break
                for r in range(h):
                    for c in range(w):
                        sr, sc = r + dr, c + dc
                        if 0 <= sr < h and 0 <= sc < w:
                            if inp[sr, sc] != out[r, c]:
                                all_match = False
                                break
                        else:
                            if out[r, c] != 0:  # assume border is 0
                                all_match = False
                                break
                    if not all_match:
                        break
                if not all_match:
                    break
            
            if all_match:
                test_inp = np.array(task['test'][0]['input'])
                th, tw = test_inp.shape
                pred = np.zeros((th, tw), dtype=int)
                for r in range(th):
                    for c in range(tw):
                        sr, sc = r + dr, c + dc
                        if 0 <= sr < th and 0 <= sc < tw:
                            pred[r, c] = test_inp[sr, sc]
                return pred.tolist()
    
    return None


# ═══════════════════════════════════════════════════════════════
# Solver 8: Output is a constant grid
# ═══════════════════════════════════════════════════════════════

def solve_constant_output(task: dict) -> Optional[list]:
    """All training outputs are the same grid."""
    outputs = [np.array(ex['output']) for ex in task['train']]
    
    for out in outputs[1:]:
        if out.shape != outputs[0].shape or not np.array_equal(out, outputs[0]):
            return None
    
    return outputs[0].tolist()


# ═══════════════════════════════════════════════════════════════
# Solver 9: Downscale by factor
# ═══════════════════════════════════════════════════════════════

def solve_downscale(task: dict) -> Optional[list]:
    """Output is input downscaled by integer factor (mode/most common)."""
    for ex in task['train']:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        ih, iw = inp.shape
        oh, ow = out.shape
        if ih % oh != 0 or iw % ow != 0:
            return None
    
    inp0 = np.array(task['train'][0]['input'])
    out0 = np.array(task['train'][0]['output'])
    fh, fw = inp0.shape[0] // out0.shape[0], inp0.shape[1] // out0.shape[1]
    
    # Check consistent factor
    for ex in task['train']:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        if inp.shape[0] // out.shape[0] != fh or inp.shape[1] // out.shape[1] != fw:
            return None
    
    # Try different aggregation methods
    for agg_name, agg_fn in [
        ('mode', lambda block: np.bincount(block.flatten().astype(int)).argmax()),
        ('max', lambda block: block.max()),
        ('min_nonzero', lambda block: block[block > 0].min() if np.any(block > 0) else 0),
        ('topleft', lambda block: block[0, 0]),
    ]:
        all_match = True
        for ex in task['train']:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            oh, ow = out.shape
            pred = np.zeros_like(out)
            for r in range(oh):
                for c in range(ow):
                    block = inp[r*fh:(r+1)*fh, c*fw:(c+1)*fw]
                    try:
                        pred[r, c] = agg_fn(block)
                    except:
                        all_match = False
                        break
                if not all_match:
                    break
            if not all_match or not np.array_equal(pred, out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            th, tw = test_inp.shape
            oh, ow = th // fh, tw // fw
            pred = np.zeros((oh, ow), dtype=int)
            for r in range(oh):
                for c in range(ow):
                    block = test_inp[r*fh:(r+1)*fh, c*fw:(c+1)*fw]
                    pred[r, c] = agg_fn(block)
            return pred.tolist()
    
    return None


# ═══════════════════════════════════════════════════════════════
# Solver 10: Upscale by factor
# ═══════════════════════════════════════════════════════════════

def solve_upscale(task: dict) -> Optional[list]:
    """Output is input upscaled by integer factor."""
    for ex in task['train']:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        oh, ow = out.shape
        ih, iw = inp.shape
        if oh % ih != 0 or ow % iw != 0:
            return None
    
    inp0 = np.array(task['train'][0]['input'])
    out0 = np.array(task['train'][0]['output'])
    fh, fw = out0.shape[0] // inp0.shape[0], out0.shape[1] // inp0.shape[1]
    
    for ex in task['train']:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        if out.shape[0] // inp.shape[0] != fh or out.shape[1] // inp.shape[1] != fw:
            return None
        pred = np.repeat(np.repeat(inp, fh, axis=0), fw, axis=1)
        if not np.array_equal(pred, out):
            return None
    
    test_inp = np.array(task['test'][0]['input'])
    return np.repeat(np.repeat(test_inp, fh, axis=0), fw, axis=1).tolist()


# ═══════════════════════════════════════════════════════════════
# Run all solvers
# ═══════════════════════════════════════════════════════════════

ALL_SOLVERS = [
    ('constant', solve_constant_output),
    ('tiling', solve_tiling),
    ('upscale', solve_upscale),
    ('downscale', solve_downscale),
    ('gravity', solve_gravity),
    ('sorting', solve_sorting),
    ('subgrid', solve_subgrid_extract),
    ('spatial_corr', solve_spatial_correspondence),
    ('majority', solve_majority_rule),
]


def solve_task(task: dict) -> Tuple[Optional[list], str]:
    """Try all heuristic solvers, return (prediction, method) or (None, 'none')."""
    for name, solver_fn in ALL_SOLVERS:
        try:
            pred = solver_fn(task)
            if pred is not None:
                return pred, name
        except Exception:
            continue
    return None, 'none'


def main():
    eval_dir = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation'
    
    # Load previously solved
    prev_solved = set()
    for f in ['arc_agi1_eval_results.json', 'arc_emerged_eval_results.json']:
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            prev_solved |= set(d.get('solved_ids', []))
    
    print(f"=== HEURISTIC SOLVER BATTERY ===")
    print(f"Previously solved: {len(prev_solved)}")
    
    solved = []
    methods = {}
    t_start = time.time()
    total = 0
    
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith('.json'):
            continue
        tid = fn.replace('.json', '')
        total += 1
        
        with open(os.path.join(eval_dir, fn)) as f:
            task = json.load(f)
        
        test_output = task['test'][0].get('output')
        if test_output is None:
            continue
        
        pred, method = solve_task(task)
        if pred is not None and grids_match(pred, test_output):
            solved.append(tid)
            methods[method] = methods.get(method, 0) + 1
            is_new = "🆕" if tid not in prev_solved else ""
            print(f"  ✅ {tid} via {method} {is_new}")
    
    elapsed = time.time() - t_start
    new_solves = [t for t in solved if t not in prev_solved]
    combined = prev_solved | set(solved)
    
    print(f"\n=== RESULTS ({elapsed:.1f}s) ===")
    print(f"Heuristic solved: {len(solved)}/{total}")
    print(f"New (not in prev): {len(new_solves)}")
    print(f"Combined total: {len(combined)}/400 ({len(combined)/4:.1f}%)")
    print(f"Methods: {methods}")
    
    if new_solves:
        print(f"\n🆕 New solves: {new_solves}")
    
    # Save
    output = {
        'heuristic_solved': len(solved),
        'new_solves': len(new_solves),
        'total': len(combined),
        'total_pct': round(len(combined) / 4, 1),
        'solved_ids': solved,
        'new_solve_ids': new_solves,
        'combined_ids': sorted(combined),
        'methods': methods,
    }
    with open('arc_heuristic_eval_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to arc_heuristic_eval_results.json")


if __name__ == '__main__':
    main()
