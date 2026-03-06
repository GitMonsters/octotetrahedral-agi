#!/usr/bin/env python3
"""
Per-pixel rule learning ARC solver.
Instead of whole-grid transforms, learns rules for individual pixels.
For each output pixel, finds which input feature(s) determine its value.
"""

import os
import sys
import json
import time
import multiprocessing
import numpy as np
from typing import Optional, Tuple, List


def grids_match(pred, target) -> bool:
    if pred is None or target is None:
        return False
    a = np.array(pred)
    b = np.array(target)
    return a.shape == b.shape and np.array_equal(a, b)


# ═══════════════════════════════════════════════════════════
# Feature extractors for a single cell
# ═══════════════════════════════════════════════════════════

def extract_features(grid: np.ndarray, r: int, c: int) -> dict:
    """Extract features for cell (r,c) in grid."""
    h, w = grid.shape
    val = int(grid[r, c])
    
    # Basic position/value features
    features = {
        'val': val,
        'r': r,
        'c': c,
        'r_from_bottom': h - 1 - r,
        'c_from_right': w - 1 - c,
    }
    
    # Neighborhood (4-connected)
    for dr, dc, name in [(-1,0,'up'), (1,0,'down'), (0,-1,'left'), (0,1,'right')]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            features[f'n_{name}'] = int(grid[nr, nc])
        else:
            features[f'n_{name}'] = -1  # border

    # 8-connected neighbors
    for dr, dc, name in [(-1,-1,'ul'), (-1,1,'ur'), (1,-1,'dl'), (1,1,'dr')]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            features[f'n_{name}'] = int(grid[nr, nc])
        else:
            features[f'n_{name}'] = -1
    
    # Count of each neighbor value
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                neighbors.append(int(grid[nr, nc]))
    
    features['n_count_same'] = sum(1 for n in neighbors if n == val)
    features['n_count_zero'] = sum(1 for n in neighbors if n == 0)
    features['n_count_nonzero'] = sum(1 for n in neighbors if n != 0)
    features['n_count_total'] = len(neighbors)
    features['n_max'] = max(neighbors) if neighbors else 0
    features['n_min'] = min(neighbors) if neighbors else 0
    
    # Row and column statistics
    row = grid[r, :]
    col = grid[:, c]
    features['row_max'] = int(row.max())
    features['row_min'] = int(row.min())
    features['row_nonzero'] = int(np.count_nonzero(row))
    features['col_max'] = int(col.max())
    features['col_min'] = int(col.min())
    features['col_nonzero'] = int(np.count_nonzero(col))
    
    # Is on border?
    features['is_top'] = int(r == 0)
    features['is_bottom'] = int(r == h - 1)
    features['is_left'] = int(c == 0)
    features['is_right'] = int(c == w - 1)
    features['is_border'] = int(r == 0 or r == h-1 or c == 0 or c == w-1)
    
    # Mirror positions
    features['val_mirror_h'] = int(grid[r, w-1-c])
    features['val_mirror_v'] = int(grid[h-1-r, c])
    features['val_mirror_hv'] = int(grid[h-1-r, w-1-c])
    
    return features


# ═══════════════════════════════════════════════════════════
# Rule templates
# ═══════════════════════════════════════════════════════════

def try_identity_rule(examples):
    """out[r,c] = in[r,c] always (identity transform)."""
    for inp, out in examples:
        if not np.array_equal(inp, out):
            return None
    return lambda inp, r, c: int(inp[r, c])


def try_single_feature_rule(examples):
    """out[r,c] = f(single_feature) for some feature and mapping f."""
    if not examples:
        return None
    
    # Collect all (feature_val, output_val) pairs across all examples and positions
    feature_names = None
    all_data = []
    
    for inp, out in examples:
        if inp.shape != out.shape:
            return None
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                feats = extract_features(inp, r, c)
                if feature_names is None:
                    feature_names = list(feats.keys())
                target = int(out[r, c])
                all_data.append((feats, target))
    
    # For each feature, check if it uniquely determines the output
    for fname in feature_names:
        mapping = {}
        consistent = True
        for feats, target in all_data:
            fval = feats[fname]
            if fval in mapping:
                if mapping[fval] != target:
                    consistent = False
                    break
            else:
                mapping[fval] = target
        
        if consistent and len(mapping) > 0:
            # Verify this isn't trivially identity
            if fname == 'val' and all(k == v for k, v in mapping.items()):
                continue
            
            return fname, mapping
    
    return None


def try_two_feature_rule(examples):
    """out[r,c] = f(feature1, feature2) for some features and mapping."""
    if not examples:
        return None
    
    # Key features to try (limited for speed)
    key_features = ['val', 'n_count_same', 'n_count_nonzero', 'is_border', 
                    'n_up', 'n_down', 'n_left', 'n_right',
                    'row_nonzero', 'col_nonzero']
    
    all_data = []
    for inp, out in examples:
        if inp.shape != out.shape:
            return None
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                feats = extract_features(inp, r, c)
                target = int(out[r, c])
                all_data.append((feats, target))
    
    for i, f1 in enumerate(key_features):
        for f2 in key_features[i+1:]:
            mapping = {}
            consistent = True
            for feats, target in all_data:
                key = (feats.get(f1), feats.get(f2))
                if key in mapping:
                    if mapping[key] != target:
                        consistent = False
                        break
                else:
                    mapping[key] = target
            
            if consistent and len(mapping) > 1:
                return f1, f2, mapping
    
    return None


def try_conditional_rules(examples):
    """if condition(cell) then output = X else output = cell_val."""
    if not examples:
        return None
    
    # Collect cells that change vs don't change
    changes = []
    no_changes = []
    
    for inp, out in examples:
        if inp.shape != out.shape:
            return None
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                feats = extract_features(inp, r, c)
                old_v = int(inp[r, c])
                new_v = int(out[r, c])
                if old_v != new_v:
                    changes.append((feats, old_v, new_v))
                else:
                    no_changes.append((feats,))
    
    if not changes:
        return None
    
    # What's the single new value for changed cells?
    new_vals = set(nv for _, _, nv in changes)
    if len(new_vals) != 1:
        return None
    new_val = new_vals.pop()
    
    # What feature distinguishes changed from unchanged?
    feature_names = list(changes[0][0].keys())
    for fname in feature_names:
        changed_vals = set(f[fname] for f, _, _ in changes)
        unchanged_vals = set(f[0][fname] for f in no_changes)
        
        # Check if there's a threshold that separates them
        if changed_vals.isdisjoint(unchanged_vals):
            return fname, changed_vals, new_val
    
    return None


# ═══════════════════════════════════════════════════════════
# Main solver
# ═══════════════════════════════════════════════════════════

def solve_pixel_rules(task: dict) -> Optional[list]:
    """Try to solve task using per-pixel rules."""
    train_pairs = [(np.array(ex['input']), np.array(ex['output'])) for ex in task['train']]
    test_input = np.array(task['test'][0]['input'])
    
    # Only same-size tasks
    for inp, out in train_pairs:
        if inp.shape != out.shape:
            return None
    if test_input.shape != train_pairs[0][0].shape:
        return None
    
    h, w = test_input.shape
    
    # Try single-feature rule
    result = try_single_feature_rule(train_pairs)
    if result is not None:
        fname, mapping = result
        pred = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                feats = extract_features(test_input, r, c)
                fval = feats[fname]
                if fval in mapping:
                    pred[r, c] = mapping[fval]
                else:
                    pred[r, c] = int(test_input[r, c])
        return pred.tolist(), f'single:{fname}'
    
    # Try two-feature rule
    result = try_two_feature_rule(train_pairs)
    if result is not None:
        f1, f2, mapping = result
        pred = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                feats = extract_features(test_input, r, c)
                key = (feats.get(f1), feats.get(f2))
                if key in mapping:
                    pred[r, c] = mapping[key]
                else:
                    pred[r, c] = int(test_input[r, c])
        return pred.tolist(), f'two:{f1}+{f2}'
    
    # Try conditional rules
    result = try_conditional_rules(train_pairs)
    if result is not None:
        fname, changed_vals, new_val = result
        pred = test_input.copy()
        for r in range(h):
            for c in range(w):
                feats = extract_features(test_input, r, c)
                if feats[fname] in changed_vals:
                    pred[r, c] = new_val
        return pred.tolist(), f'cond:{fname}→{new_val}'
    
    return None, 'none'


def _worker(task_json, result_queue):
    """Worker for subprocess."""
    task = json.loads(task_json)
    test_output = task['test'][0].get('output')
    if test_output is None:
        result_queue.put((False, 'no_gt'))
        return
    
    result = solve_pixel_rules(task)
    if result is not None:
        pred, method = result
        if pred is not None and grids_match(pred, test_output):
            result_queue.put((True, method))
            return
    
    result_queue.put((False, 'none'))


def solve_with_timeout(task_json, timeout_s=10):
    """Run in forked subprocess."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(task_json, q))
    p.start()
    p.join(timeout=timeout_s)
    
    if p.is_alive():
        p.terminate()
        p.join(timeout=2)
        return False, 'timeout'
    
    if not q.empty():
        try:
            return q.get_nowait()
        except Exception:
            pass
    return False, 'error'


def main():
    eval_dir = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation'
    
    prev_solved = set()
    for f in ['arc_agi1_eval_results.json', 'arc_emerged_eval_results.json']:
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            prev_solved |= set(d.get('solved_ids', []))

    print(f"=== PIXEL RULE SOLVER — ARC-AGI-1 EVAL ===")
    print(f"Previously solved: {len(prev_solved)}")

    solved = []
    methods = {}
    total = 0
    t_start = time.time()

    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith('.json'):
            continue
        tid = fn.replace('.json', '')
        total += 1
        
        with open(os.path.join(eval_dir, fn)) as f:
            task = json.load(f)

        is_solved, method = solve_with_timeout(json.dumps(task), timeout_s=15)
        
        if is_solved:
            solved.append(tid)
            methods[method] = methods.get(method, 0) + 1
            is_new = "🆕" if tid not in prev_solved else ""
            print(f"  ✅ {tid} via {method} {is_new}")

    elapsed = time.time() - t_start
    new_solves = [t for t in solved if t not in prev_solved]
    combined = prev_solved | set(solved)

    print(f"\n=== RESULTS ({elapsed:.0f}s) ===")
    print(f"Pixel-rule solved: {len(solved)}/{total}")
    print(f"New (not in prev): {len(new_solves)}")
    print(f"Combined total: {len(combined)}/400 ({len(combined)/4:.1f}%)")
    print(f"Methods: {methods}")

    if new_solves:
        print(f"\n🆕 New: {new_solves}")

    output = {
        'pixel_rule_solved': len(solved),
        'new_solves': len(new_solves),
        'total': len(combined),
        'total_pct': round(len(combined) / 4, 1),
        'solved_ids': solved,
        'new_solve_ids': new_solves,
        'combined_ids': sorted(combined),
        'methods': methods,
    }
    with open('arc_pixel_rule_eval_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to arc_pixel_rule_eval_results.json")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
