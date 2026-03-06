#!/usr/bin/env python3
"""
Extended ARC solver: more strategies for unsolved tasks.
Builds on arc_advanced_solver findings, adds cellular automata,
connected-component analysis, symmetry completion, and more.
"""

import os, sys, json, time, multiprocessing
import numpy as np
from collections import Counter, defaultdict
from scipy import ndimage


def grids_match(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and np.array_equal(a, b)

def find_bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


# ═══════════════════════════════════════════════
# 1: Cellular Automata (iterative local rules)
# ═══════════════════════════════════════════════

def apply_ca_rule(grid, rule_fn, bg=0, max_steps=50):
    """Apply a local rule iteratively until convergence."""
    h, w = grid.shape
    current = grid.copy()
    for step in range(max_steps):
        new = current.copy()
        changed = False
        for r in range(h):
            for c in range(w):
                old_val = int(current[r, c])
                n4 = {}
                for name, (dr, dc) in [('u',(-1,0)),('d',(1,0)),('l',(0,-1)),('r',(0,1))]:
                    nr, nc = r+dr, c+dc
                    n4[name] = int(current[nr, nc]) if 0 <= nr < h and 0 <= nc < w else -1
                
                new_val = rule_fn(old_val, n4, bg, r, c, h, w, step)
                if new_val != old_val:
                    new[r, c] = new_val
                    changed = True
        current = new
        if not changed:
            break
    return current


def solve_cellular_automata(task):
    """Try various CA rules."""
    train = task['train']
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    bg = find_bg(np.array(train[0]['input']))
    
    # Rule: fill bg cells that have exactly N non-bg neighbors
    for n_thresh in [1, 2, 3, 4]:
        def make_rule(thresh):
            def rule(val, n4, bg, r, c, h, w, step):
                if val != bg:
                    return val
                adj = [v for v in n4.values() if v != -1 and v != bg]
                if len(adj) >= thresh:
                    return Counter(adj).most_common(1)[0][0]
                return val
            return rule
        
        rule = make_rule(n_thresh)
        all_match = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            pred = apply_ca_rule(inp, rule, bg)
            if not np.array_equal(pred, out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            return apply_ca_rule(test_inp, rule, bg).tolist(), f'ca_fill_{n_thresh}'
    
    # Rule: remove isolated cells (cells with no same-color neighbor)
    def remove_isolated(val, n4, bg, r, c, h, w, step):
        if val == bg:
            return val
        adj_same = sum(1 for v in n4.values() if v == val)
        if adj_same == 0:
            return bg
        return val
    
    all_match = True
    for ex in train:
        pred = apply_ca_rule(np.array(ex['input']), remove_isolated, bg, max_steps=1)
        if not np.array_equal(pred, np.array(ex['output'])):
            all_match = False
            break
    if all_match:
        return apply_ca_rule(np.array(task['test'][0]['input']), remove_isolated, bg, max_steps=1).tolist(), 'ca_remove_isolated'
    
    return None


# ═══════════════════════════════════════════════
# 2: Connected Component Analysis
# ═══════════════════════════════════════════════

def get_objects(grid, bg=0):
    mask = grid != bg
    if not mask.any():
        return []
    labeled, n = ndimage.label(mask)
    objs = []
    for i in range(1, n+1):
        m = labeled == i
        rows, cols = np.where(m)
        colors = Counter(grid[m].flatten().tolist())
        objs.append({
            'id': i, 'mask': m, 'size': int(m.sum()),
            'bbox': (rows.min(), cols.min(), rows.max()+1, cols.max()+1),
            'primary_color': colors.most_common(1)[0][0],
            'n_colors': len(colors),
            'width': cols.max()-cols.min()+1,
            'height': rows.max()-rows.min()+1,
        })
    return objs


def solve_object_filter(task):
    """Output = specific objects from input based on property."""
    train = task['train']
    
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        if inp.shape != out.shape:
            return None
    
    bg = find_bg(np.array(train[0]['input']))
    
    # What objects are kept vs removed?
    # Try: keep only objects of specific size / color / shape
    filters = [
        ('keep_max_size', lambda objs: [max(objs, key=lambda o: o['size'])] if objs else []),
        ('keep_min_size', lambda objs: [min(objs, key=lambda o: o['size'])] if objs else []),
        ('keep_multicolor', lambda objs: [o for o in objs if o['n_colors'] > 1]),
        ('keep_singlecolor', lambda objs: [o for o in objs if o['n_colors'] == 1]),
    ]
    
    # Also try: keep objects of specific color
    inp0 = np.array(train[0]['input'])
    colors = set(inp0.flatten().tolist()) - {bg}
    for c in colors:
        filters.append((f'keep_color_{c}', lambda objs, c=c: [o for o in objs if o['primary_color'] == c]))
    
    for fname, ffn in filters:
        all_match = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            objs = get_objects(inp, bg)
            kept = ffn(objs)
            
            pred = np.full_like(inp, bg)
            for o in kept:
                pred[o['mask']] = inp[o['mask']]
            
            if not np.array_equal(pred, out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            objs = get_objects(test_inp, bg)
            kept = ffn(objs)
            pred = np.full_like(test_inp, bg)
            for o in kept:
                pred[o['mask']] = test_inp[o['mask']]
            return pred.tolist(), f'obj:{fname}'
    
    return None


def solve_object_count_output(task):
    """Output encodes object count/property in grid form."""
    train = task['train']
    out0 = np.array(train[0]['output'])
    bg = find_bg(np.array(train[0]['input']))
    
    # 1xN or Nx1 output where each cell = property of an object
    if out0.shape[0] != 1 and out0.shape[1] != 1:
        return None
    
    n = max(out0.shape)
    
    # Check: output length = number of objects
    for ex in train:
        objs = get_objects(np.array(ex['input']), bg)
        out = np.array(ex['output']).flatten()
        if len(objs) != len(out):
            return None
    
    # Try: output[i] = size/color/width of i-th object (sorted somehow)
    sort_keys = [
        ('by_row', lambda o: o['bbox'][0]),
        ('by_col', lambda o: o['bbox'][1]),
        ('by_size_asc', lambda o: o['size']),
        ('by_size_desc', lambda o: -o['size']),
    ]
    
    value_extractors = [
        ('color', lambda o: o['primary_color']),
        ('size', lambda o: o['size']),
        ('width', lambda o: o['width']),
        ('height', lambda o: o['height']),
    ]
    
    for sk_name, sk_fn in sort_keys:
        for ve_name, ve_fn in value_extractors:
            all_match = True
            for ex in train:
                objs = get_objects(np.array(ex['input']), bg)
                objs_sorted = sorted(objs, key=sk_fn)
                out = np.array(ex['output']).flatten()
                
                pred = [ve_fn(o) for o in objs_sorted]
                if pred != list(out):
                    all_match = False
                    break
            
            if all_match:
                test_inp = np.array(task['test'][0]['input'])
                objs = get_objects(test_inp, bg)
                objs_sorted = sorted(objs, key=sk_fn)
                pred = [ve_fn(o) for o in objs_sorted]
                
                if out0.shape[0] == 1:
                    return [pred], f'objcount:{sk_name}:{ve_name}'
                else:
                    return [[v] for v in pred], f'objcount:{sk_name}:{ve_name}'
    
    return None


# ═══════════════════════════════════════════════
# 3: Symmetry Completion
# ═══════════════════════════════════════════════

def solve_symmetry(task):
    """Complete partial patterns using detected symmetry."""
    train = task['train']
    
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    bg = find_bg(np.array(train[0]['input']))
    
    # Try horizontal mirror completion
    symmetries = [
        ('hmirror', lambda g, r, c: (r, g.shape[1]-1-c)),
        ('vmirror', lambda g, r, c: (g.shape[0]-1-r, c)),
        ('dmirror', lambda g, r, c: (g.shape[0]-1-r, g.shape[1]-1-c)),
    ]
    
    for sym_name, sym_fn in symmetries:
        all_match = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            h, w = inp.shape
            
            pred = inp.copy()
            for r in range(h):
                for c in range(w):
                    if pred[r, c] == bg:
                        mr, mc = sym_fn(pred, r, c)
                        if 0 <= mr < h and 0 <= mc < w and pred[mr, mc] != bg:
                            pred[r, c] = pred[mr, mc]
            
            if not np.array_equal(pred, out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            h, w = test_inp.shape
            pred = test_inp.copy()
            for r in range(h):
                for c in range(w):
                    if pred[r, c] == bg:
                        mr, mc = sym_fn(pred, r, c)
                        if 0 <= mr < h and 0 <= mc < w and pred[mr, mc] != bg:
                            pred[r, c] = pred[mr, mc]
            return pred.tolist(), f'sym:{sym_name}'
    
    return None


# ═══════════════════════════════════════════════
# 4: Color Mapping Chains
# ═══════════════════════════════════════════════

def solve_color_rotation(task):
    """Each color shifts by some fixed amount."""
    train = task['train']
    
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    # Try: each non-bg color C → C+offset (mod 10)
    bg = find_bg(np.array(train[0]['input']))
    
    for offset in range(1, 10):
        cmap = {}
        for c in range(10):
            if c == bg:
                cmap[c] = bg
            else:
                cmap[c] = (c + offset) % 10
                if cmap[c] == bg:
                    cmap[c] = (cmap[c] + 1) % 10
        
        all_match = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            pred = np.vectorize(lambda x: cmap.get(x, x))(inp)
            if not np.array_equal(pred, out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            pred = np.vectorize(lambda x: cmap.get(x, x))(test_inp)
            return pred.tolist(), f'color_rot_{offset}'
    
    # Try: learn per-color map from first example
    inp0, out0 = np.array(train[0]['input']), np.array(train[0]['output'])
    if inp0.shape != out0.shape:
        return None
    
    cmap = {}
    for iv, ov in zip(inp0.flatten(), out0.flatten()):
        iv, ov = int(iv), int(ov)
        if iv in cmap:
            if cmap[iv] != ov:
                return None
        else:
            cmap[iv] = ov
    
    # Verify on other examples
    for ex in train[1:]:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        if inp.shape != out.shape:
            return None
        for iv, ov in zip(inp.flatten(), out.flatten()):
            iv, ov = int(iv), int(ov)
            if iv in cmap:
                if cmap[iv] != ov:
                    return None
            else:
                cmap[iv] = ov
    
    test_inp = np.array(task['test'][0]['input'])
    # Check all values are in map
    for v in test_inp.flatten():
        if int(v) not in cmap:
            return None
    
    pred = np.vectorize(lambda x: cmap[int(x)])(test_inp)
    return pred.tolist(), 'color_map'


# ═══════════════════════════════════════════════
# 5: Tile / Repeat Input
# ═══════════════════════════════════════════════

def solve_tile_repeat(task):
    """Output = input tiled/repeated in some pattern."""
    train = task['train']
    
    ex0 = train[0]
    inp, out = np.array(ex0['input']), np.array(ex0['output'])
    
    ih, iw = inp.shape
    oh, ow = out.shape
    
    if oh % ih != 0 or ow % iw != 0:
        return None
    
    rh, rw = oh // ih, ow // iw
    if rh == 1 and rw == 1:
        return None
    
    # Check if output = tile(input, rh, rw)
    tiled = np.tile(inp, (rh, rw))
    if not np.array_equal(tiled, out):
        return None
    
    # Verify on all
    for ex in train[1:]:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        ih2, iw2 = inp.shape
        oh2, ow2 = out.shape
        if oh2 // ih2 != rh or ow2 // iw2 != rw:
            return None
        if not np.array_equal(np.tile(inp, (rh, rw)), out):
            return None
    
    test_inp = np.array(task['test'][0]['input'])
    return np.tile(test_inp, (rh, rw)).tolist(), f'tile_{rh}x{rw}'


# ═══════════════════════════════════════════════
# 6: Border / Frame operations
# ═══════════════════════════════════════════════

def solve_border_ops(task):
    """Add/remove border, or extract interior."""
    train = task['train']
    
    ex0 = train[0]
    inp, out = np.array(ex0['input']), np.array(ex0['output'])
    ih, iw = inp.shape
    oh, ow = out.shape
    
    # Output = input interior (remove 1-cell border)
    for border in [1, 2]:
        if ih - 2*border == oh and iw - 2*border == ow:
            interior = inp[border:ih-border, border:iw-border]
            if np.array_equal(interior, out):
                all_match = True
                for ex in train[1:]:
                    i, o = np.array(ex['input']), np.array(ex['output'])
                    pred = i[border:i.shape[0]-border, border:i.shape[1]-border]
                    if not np.array_equal(pred, o):
                        all_match = False
                        break
                
                if all_match:
                    t = np.array(task['test'][0]['input'])
                    return t[border:t.shape[0]-border, border:t.shape[1]-border].tolist(), f'remove_border_{border}'
    
    return None


# ═══════════════════════════════════════════════
# 7: Replace color with pattern
# ═══════════════════════════════════════════════

def solve_replace_with_pattern(task):
    """Replace each instance of a pattern/color with another."""
    train = task['train']
    
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    inp0, out0 = np.array(train[0]['input']), np.array(train[0]['output'])
    bg = find_bg(inp0)
    h, w = inp0.shape
    
    # Find changed cells
    diff = inp0 != out0
    if not diff.any():
        return None
    
    # Are all changes in a consistent local pattern around specific marker cells?
    # For each changed cell, what's its relationship to nearby non-bg cells?
    
    # Simple case: single color replacement with neighbor-dependent new color
    changed_from = set(inp0[diff].flatten().tolist())
    
    if len(changed_from) == 1:
        src_color = list(changed_from)[0]
        # All cells of src_color changed — what determines the new color?
        # Try: new color = nearest non-bg, non-src color
        pass
    
    return None


# ═══════════════════════════════════════════════
# 8: Max/Min row/col operations
# ═══════════════════════════════════════════════

def solve_row_col_aggregation(task):
    """Output is aggregation of rows/cols (max, min, OR, AND)."""
    train = task['train']
    
    out0 = np.array(train[0]['output'])
    inp0 = np.array(train[0]['input'])
    bg = find_bg(inp0)
    
    # Output is 1 row: OR of all input rows
    if out0.shape[0] == 1 and out0.shape[1] == inp0.shape[1]:
        # Try row-wise OR (any non-bg → that color)
        def row_or(inp):
            h, w = inp.shape
            bg_i = find_bg(inp)
            result = np.full((1, w), bg_i, dtype=int)
            for c in range(w):
                col = inp[:, c]
                non_bg = col[col != bg_i]
                if len(non_bg) > 0:
                    result[0, c] = Counter(non_bg.tolist()).most_common(1)[0][0]
            return result
        
        all_match = True
        for ex in train:
            pred = row_or(np.array(ex['input']))
            if not np.array_equal(pred, np.array(ex['output'])):
                all_match = False
                break
        
        if all_match:
            return row_or(np.array(task['test'][0]['input'])).tolist(), 'row_or'
    
    # Output is 1 col: OR of all input cols
    if out0.shape[1] == 1 and out0.shape[0] == inp0.shape[0]:
        def col_or(inp):
            h, w = inp.shape
            bg_i = find_bg(inp)
            result = np.full((h, 1), bg_i, dtype=int)
            for r in range(h):
                row = inp[r, :]
                non_bg = row[row != bg_i]
                if len(non_bg) > 0:
                    result[r, 0] = Counter(non_bg.tolist()).most_common(1)[0][0]
            return result
        
        all_match = True
        for ex in train:
            pred = col_or(np.array(ex['input']))
            if not np.array_equal(pred, np.array(ex['output'])):
                all_match = False
                break
        
        if all_match:
            return col_or(np.array(task['test'][0]['input'])).tolist(), 'col_or'
    
    return None


# ═══════════════════════════════════════════════
# 9: Diagonal operations
# ═══════════════════════════════════════════════

def solve_diagonal_ops(task):
    """Extract or manipulate diagonals."""
    train = task['train']
    
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        if inp.shape != out.shape:
            return None
    
    bg = find_bg(np.array(train[0]['input']))
    
    # Try: reflect across main diagonal
    all_match = True
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        if inp.shape[0] != inp.shape[1]:
            all_match = False
            break
        # Max of cell and its diagonal mirror
        pred = inp.copy()
        h = inp.shape[0]
        for r in range(h):
            for c in range(h):
                if inp[r, c] != bg and inp[c, r] == bg:
                    pred[c, r] = inp[r, c]
                elif inp[c, r] != bg and inp[r, c] == bg:
                    pred[r, c] = inp[c, r]
        if not np.array_equal(pred, out):
            all_match = False
            break
    
    if all_match:
        test_inp = np.array(task['test'][0]['input'])
        if test_inp.shape[0] == test_inp.shape[1]:
            h = test_inp.shape[0]
            pred = test_inp.copy()
            for r in range(h):
                for c in range(h):
                    if test_inp[r, c] != bg and test_inp[c, r] == bg:
                        pred[c, r] = test_inp[r, c]
                    elif test_inp[c, r] != bg and test_inp[r, c] == bg:
                        pred[r, c] = test_inp[c, r]
            return pred.tolist(), 'diag_mirror'
    
    return None


# ═══════════════════════════════════════════════
# 10: Grid Boolean / Set Operations
# ═══════════════════════════════════════════════

def solve_grid_boolean(task):
    """Output = boolean operation on two halves of the input."""
    train = task['train']
    
    inp0 = np.array(train[0]['input'])
    out0 = np.array(train[0]['output'])
    ih, iw = inp0.shape
    oh, ow = out0.shape
    
    bg = find_bg(inp0)
    
    # Split input in half horizontally
    splits = []
    if ih % 2 == 0 and oh == ih // 2 and ow == iw:
        splits.append(('hsplit', lambda g: (g[:g.shape[0]//2], g[g.shape[0]//2:])))
    if iw % 2 == 0 and oh == ih and ow == iw // 2:
        splits.append(('vsplit', lambda g: (g[:, :g.shape[1]//2], g[:, g.shape[1]//2:])))
    if ih == oh and iw == ow:
        # Maybe split by separator line
        for r in range(1, ih-1):
            row = inp0[r]
            if len(set(row.flatten())) == 1 and row[0] != bg:
                top = inp0[:r]
                bot = inp0[r+1:]
                if top.shape == bot.shape and top.shape == out0.shape:
                    splits.append((f'sep_h_{r}', lambda g, r=r: (g[:r], g[r+1:])))
    
    ops = [
        ('or', lambda a, b, bg: np.where((a != bg) | (b != bg), np.where(a != bg, a, b), bg)),
        ('and', lambda a, b, bg: np.where((a != bg) & (b != bg), a, bg)),
        ('xor', lambda a, b, bg: np.where((a != bg) ^ (b != bg), np.where(a != bg, a, b), bg)),
        ('diff', lambda a, b, bg: np.where((a != bg) & (b == bg), a, bg)),
    ]
    
    for split_name, split_fn in splits:
        for op_name, op_fn in ops:
            all_match = True
            for ex in train:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                try:
                    a, b = split_fn(inp)
                    pred = op_fn(a, b, bg)
                    if pred.shape != out.shape or not np.array_equal(pred, out):
                        all_match = False
                        break
                except Exception:
                    all_match = False
                    break
            
            if all_match:
                test_inp = np.array(task['test'][0]['input'])
                a, b = split_fn(test_inp)
                pred = op_fn(a, b, bg)
                return pred.tolist(), f'bool:{split_name}:{op_name}'
    
    return None


# ═══════════════════════════════════════════════
# Main solver
# ═══════════════════════════════════════════════

def solve_task(task):
    strategies = [
        solve_color_rotation,
        solve_cellular_automata,
        solve_object_filter,
        solve_object_count_output,
        solve_symmetry,
        solve_tile_repeat,
        solve_border_ops,
        solve_row_col_aggregation,
        solve_diagonal_ops,
        solve_grid_boolean,
    ]
    
    for fn in strategies:
        try:
            result = fn(task)
            if result is not None:
                return result
        except Exception:
            continue
    return None


def _worker(task_json, result_queue):
    task = json.loads(task_json)
    gt = task['test'][0].get('output')
    if gt is None:
        result_queue.put((False, 'no_gt'))
        return
    result = solve_task(task)
    if result is not None:
        pred, method = result
        if grids_match(pred, gt):
            result_queue.put((True, method))
            return
    result_queue.put((False, 'none'))


def solve_w_timeout(task_json, timeout=30):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(task_json, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join(3)
        return False, 'timeout'
    if not q.empty():
        try:
            return q.get_nowait()
        except Exception:
            pass
    return False, 'error'


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    
    eval_dir = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation'
    
    # Load all previously solved
    prev = set()
    for f in ['arc_agi1_eval_results.json', 'arc_emerged_eval_results.json', 'arc_advanced_eval_results.json']:
        if os.path.exists(f):
            d = json.load(open(f))
            prev |= set(d.get('solved_ids', d.get('combined_ids', [])))
    
    log.info("=" * 60)
    log.info("🔬 EXTENDED ARC SOLVER")
    log.info(f"Previously solved: {len(prev)}/400")
    log.info("=" * 60)
    
    solved = []
    methods = {}
    total = 0
    t0 = time.time()
    
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith('.json'):
            continue
        tid = fn.replace('.json', '')
        total += 1
        
        task = json.load(open(os.path.join(eval_dir, fn)))
        ok, method = solve_w_timeout(json.dumps(task), timeout=30)
        
        if ok:
            solved.append(tid)
            methods[method] = methods.get(method, 0) + 1
            tag = "🆕" if tid not in prev else ""
            log.info(f"  ✅ {tid} [{method}] {tag}")
        
        if total % 100 == 0:
            wall = time.time() - t0
            new = len([t for t in solved if t not in prev])
            log.info(f"  [{total}/400] {len(solved)} solved (+{new} new) | {wall:.0f}s")
    
    wall = time.time() - t0
    new_ids = [t for t in solved if t not in prev]
    combined = prev | set(solved)
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 EXTENDED SOLVER RESULTS")
    log.info(f"{'='*60}")
    log.info(f"  This solver:    {len(solved)}/{total}")
    log.info(f"  🆕 New:         {len(new_ids)}")
    log.info(f"  ★ COMBINED:     {len(combined)}/400 ({len(combined)/4:.1f}%)")
    log.info(f"  Time:           {wall:.0f}s")
    log.info(f"  Methods:        {methods}")
    if new_ids:
        log.info(f"  🆕 New IDs:     {sorted(new_ids)}")
    
    out = {
        'solver_solved': len(solved),
        'new': len(new_ids),
        'combined': len(combined),
        'pct': round(len(combined)/4, 1),
        'solved_ids': sorted(solved),
        'new_ids': sorted(new_ids),
        'combined_ids': sorted(combined),
        'methods': methods,
    }
    with open('arc_extended_eval_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f"  → Saved arc_extended_eval_results.json")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
