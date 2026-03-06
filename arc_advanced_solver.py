#!/usr/bin/env python3
"""
Advanced ARC solver: local neighborhood rules, conditional transforms, 
grid partitioning, and template-based program synthesis.
"""

import os, sys, json, time, multiprocessing
import numpy as np
from collections import Counter, defaultdict
from itertools import product


def grids_match(a, b) -> bool:
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and np.array_equal(a, b)


def find_bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


# ═══════════════════════════════════════════════
# Strategy 1: Neighborhood Lookup Table
# ═══════════════════════════════════════════════

def get_neighborhood(grid, r, c, radius=1):
    """Get flattened neighborhood values (padded with -1)."""
    h, w = grid.shape
    vals = []
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                vals.append(int(grid[nr, nc]))
            else:
                vals.append(-1)  # border sentinel
    return tuple(vals)


def solve_neighborhood_lut(task, radius=1):
    """Learn output color as function of input neighborhood."""
    train = task['train']
    
    # Check all same-size
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    # Build lookup table from all training examples
    lut = {}
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                key = get_neighborhood(inp, r, c, radius)
                val = int(out[r, c])
                if key in lut:
                    if lut[key] != val:
                        return None  # inconsistent
                else:
                    lut[key] = val
    
    # Apply to test
    test_inp = np.array(task['test'][0]['input'])
    h, w = test_inp.shape
    result = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            key = get_neighborhood(test_inp, r, c, radius)
            if key not in lut:
                return None  # unseen pattern
            result[r, c] = lut[key]
    
    return result.tolist()


# ═══════════════════════════════════════════════
# Strategy 2: Abstract Neighborhood Features
# ═══════════════════════════════════════════════

def get_abstract_features(grid, r, c):
    """Extract abstract features of a cell's neighborhood."""
    h, w = grid.shape
    val = int(grid[r, c])
    bg = find_bg(grid)
    
    # Direct neighbors
    neighbors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w:
            neighbors.append(int(grid[nr, nc]))
    
    # Diagonal neighbors
    diag = []
    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w:
            diag.append(int(grid[nr, nc]))
    
    all_8 = neighbors + diag
    
    features = []
    
    # Feature sets (each is a tuple we can hash)
    # F0: self value
    features.append(('self', val))
    # F1: self + number of non-bg neighbors
    features.append(('self_nnb', val, sum(1 for n in all_8 if n != bg)))
    # F2: self + set of neighbor colors
    features.append(('self_ncols', val, tuple(sorted(set(neighbors)))))
    # F3: self + is_on_border
    features.append(('self_border', val, r==0 or r==h-1 or c==0 or c==w-1))
    # F4: self + has same-color neighbor
    features.append(('self_adj_same', val, any(n == val for n in neighbors)))
    # F5: self + count of same-color in 4-neighbors
    features.append(('self_adj_cnt', val, sum(1 for n in neighbors if n == val)))
    # F6: self + majority color in 8-neighborhood
    features.append(('self_maj', val, Counter(all_8).most_common(1)[0][0] if all_8 else bg))
    # F7: row/col parity + self
    features.append(('parity', val, r % 2, c % 2))
    # F8: distance to nearest non-bg (binned)
    features.append(('self_bg_adj', val, val == bg, sum(1 for n in neighbors if n != bg)))
    
    return features


def solve_abstract_features(task):
    """Learn output color from abstract input features."""
    train = task['train']
    
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    # Try each feature type
    n_feature_types = 9
    
    for fi in range(n_feature_types):
        lut = {}
        consistent = True
        
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            h, w = inp.shape
            for r in range(h):
                for c in range(w):
                    features = get_abstract_features(inp, r, c)
                    key = features[fi]
                    val = int(out[r, c])
                    if key in lut:
                        if lut[key] != val:
                            consistent = False
                            break
                    else:
                        lut[key] = val
                if not consistent:
                    break
            if not consistent:
                break
        
        if consistent:
            # Apply to test
            test_inp = np.array(task['test'][0]['input'])
            h, w = test_inp.shape
            result = np.zeros((h, w), dtype=int)
            complete = True
            for r in range(h):
                for c in range(w):
                    features = get_abstract_features(test_inp, r, c)
                    key = features[fi]
                    if key not in lut:
                        complete = False
                        break
                    result[r, c] = lut[key]
                if not complete:
                    break
            
            if complete:
                return result.tolist()
    
    return None


# ═══════════════════════════════════════════════
# Strategy 3: Grid Partition + Sub-transforms
# ═══════════════════════════════════════════════

def find_grid_separators(grid, bg=0):
    """Find rows/cols that act as separators (uniform color, different from content)."""
    h, w = grid.shape
    
    sep_rows = []
    for r in range(h):
        row = grid[r]
        vals = set(row.flatten())
        if len(vals) == 1 and list(vals)[0] != bg:
            sep_rows.append((r, list(vals)[0]))
    
    sep_cols = []
    for c in range(w):
        col = grid[:, c]
        vals = set(col.flatten())
        if len(vals) == 1 and list(vals)[0] != bg:
            sep_cols.append((c, list(vals)[0]))
    
    return sep_rows, sep_cols


def get_subgrids(grid, sep_rows, sep_cols):
    """Split grid into subgrids by separator lines."""
    h, w = grid.shape
    rows = [-1] + [r for r, _ in sep_rows] + [h]
    cols = [-1] + [c for c, _ in sep_cols] + [w]
    
    subgrids = []
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
            r1, r2 = rows[i]+1, rows[i+1]
            c1, c2 = cols[j]+1, cols[j+1]
            if r1 < r2 and c1 < c2:
                subgrids.append(((r1, c1, r2, c2), grid[r1:r2, c1:c2].copy()))
    
    return subgrids


def solve_grid_partition(task):
    """Split grids by separator lines and analyze subgrid patterns."""
    train = task['train']
    
    # Check if grid has separators
    ex0 = np.array(train[0]['input'])
    sr, sc = find_grid_separators(ex0)
    if not sr and not sc:
        return None
    
    # This is complex — for now just try the "pick subgrid" approach
    # Output = one of the subgrids based on some property
    out0 = np.array(train[0]['output'])
    subgrids0 = get_subgrids(ex0, sr, sc)
    
    if not subgrids0:
        return None
    
    # Check if output matches one of the subgrids
    for idx, (bbox, sg) in enumerate(subgrids0):
        if grids_match(sg, out0):
            # Verify: same index subgrid for all training examples
            all_match = True
            for ex in train[1:]:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                sr_i, sc_i = find_grid_separators(inp)
                sgs = get_subgrids(inp, sr_i, sc_i)
                if idx >= len(sgs) or not grids_match(sgs[idx][1], out):
                    all_match = False
                    break
            
            if all_match:
                test_inp = np.array(task['test'][0]['input'])
                sr_t, sc_t = find_grid_separators(test_inp)
                sgs_t = get_subgrids(test_inp, sr_t, sc_t)
                if idx < len(sgs_t):
                    return sgs_t[idx][1].tolist()
    
    # Check if output matches subgrid with special property
    # E.g., the subgrid with most non-bg cells, or most unique colors
    properties = [
        ('most_nonbg', lambda sg, bg: np.count_nonzero(sg != bg)),
        ('least_nonbg', lambda sg, bg: -np.count_nonzero(sg != bg)),
        ('most_colors', lambda sg, bg: len(set(sg.flatten()) - {bg})),
        ('least_colors', lambda sg, bg: -len(set(sg.flatten()) - {bg})),
    ]
    
    bg = find_bg(ex0)
    
    for prop_name, prop_fn in properties:
        # Find which subgrid matches output
        matching_prop = None
        for bbox, sg in subgrids0:
            if grids_match(sg, out0):
                matching_prop = prop_fn(sg, bg)
                break
        
        if matching_prop is None:
            continue
        
        # Check: is this the max/min for this property?
        prop_vals = [(prop_fn(sg, bg), i) for i, (_, sg) in enumerate(subgrids0)]
        prop_vals.sort(reverse=True)
        target_idx = None
        for i, (_, sg) in enumerate(subgrids0):
            if grids_match(sg, out0):
                # Is this the max?
                if prop_vals[0][1] == i:
                    target_idx = 'max'
                elif prop_vals[-1][1] == i:
                    target_idx = 'min'
                break
        
        if target_idx is None:
            continue
        
        # Verify on all training examples
        all_match = True
        for ex in train[1:]:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            bg_i = find_bg(inp)
            sr_i, sc_i = find_grid_separators(inp)
            sgs = get_subgrids(inp, sr_i, sc_i)
            
            if not sgs:
                all_match = False
                break
            
            pvals = [(prop_fn(sg, bg_i), j) for j, (_, sg) in enumerate(sgs)]
            pvals.sort(reverse=True)
            
            if target_idx == 'max':
                best = pvals[0][1]
            else:
                best = pvals[-1][1]
            
            if not grids_match(sgs[best][1], out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            bg_t = find_bg(test_inp)
            sr_t, sc_t = find_grid_separators(test_inp)
            sgs_t = get_subgrids(test_inp, sr_t, sc_t)
            
            if sgs_t:
                pvals = [(prop_fn(sg, bg_t), j) for j, (_, sg) in enumerate(sgs_t)]
                pvals.sort(reverse=True)
                if target_idx == 'max':
                    return sgs_t[pvals[0][1]][1].tolist()
                else:
                    return sgs_t[pvals[-1][1]][1].tolist()
    
    return None


# ═══════════════════════════════════════════════
# Strategy 4: Conditional Color Rules
# ═══════════════════════════════════════════════

def solve_conditional_color(task):
    """Find rules: color X → Y when condition holds."""
    train = task['train']
    
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    # Gather all cell changes
    changes = []
    no_changes = []
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        h, w = inp.shape
        bg = find_bg(inp)
        for r in range(h):
            for c in range(w):
                if inp[r, c] != out[r, c]:
                    # Cell changed
                    # Gather context
                    n4 = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            n4.append(int(inp[nr, nc]))
                    
                    changes.append({
                        'from': int(inp[r, c]),
                        'to': int(out[r, c]),
                        'adj_colors': set(n4),
                        'is_bg': inp[r, c] == bg,
                        'adj_has_bg': bg in n4,
                        'n_adj_same': sum(1 for n in n4 if n == inp[r, c]),
                    })
    
    if not changes:
        return None
    
    # Rule type 1: "all cells of color X become color Y"
    color_rules = {}
    for ch in changes:
        key = ch['from']
        if key in color_rules:
            if color_rules[key] != ch['to']:
                color_rules = None
                break
        else:
            color_rules[key] = ch['to']
    
    if color_rules is not None:
        # Verify no unchanged cells contradict the rule
        valid = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    v = int(inp[r, c])
                    if v in color_rules and int(out[r, c]) != color_rules[v]:
                        valid = False
                        break
                    if v not in color_rules and int(out[r, c]) != v:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            test_inp = np.array(task['test'][0]['input'])
            result = test_inp.copy()
            for old, new in color_rules.items():
                result[test_inp == old] = new
            return result.tolist()
    
    # Rule type 2: "bg cells adjacent to color X become color Y"
    bg = find_bg(np.array(train[0]['input']))
    bg_changes = [ch for ch in changes if ch['is_bg']]
    if bg_changes and all(ch['is_bg'] for ch in changes):
        # All changes are bg → something. What determines the color?
        # Try: adjacent non-bg color determines fill color
        rule = {}  # adj_color → fill_color
        for ch in bg_changes:
            adj_nonbg = ch['adj_colors'] - {bg}
            if len(adj_nonbg) == 1:
                key = list(adj_nonbg)[0]
                if key in rule:
                    if rule[key] != ch['to']:
                        rule = None
                        break
                else:
                    rule[key] = ch['to']
        
        if rule is not None and rule:
            # Verify
            valid = True
            for ex in train:
                inp, out = np.array(ex['input']), np.array(ex['output'])
                h, w = inp.shape
                for r in range(h):
                    for c in range(w):
                        if inp[r, c] == bg and out[r, c] != bg:
                            n4 = []
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    n4.append(int(inp[nr, nc]))
                            adj_nb = set(n4) - {bg}
                            if len(adj_nb) == 1 and list(adj_nb)[0] in rule:
                                if rule[list(adj_nb)[0]] != int(out[r, c]):
                                    valid = False
                                    break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                test_inp = np.array(task['test'][0]['input'])
                result = test_inp.copy()
                h, w = result.shape
                for r in range(h):
                    for c in range(w):
                        if result[r, c] == bg:
                            n4 = []
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    n4.append(int(test_inp[nr, nc]))
                            adj_nb = set(n4) - {bg}
                            if len(adj_nb) == 1 and list(adj_nb)[0] in rule:
                                result[r, c] = rule[list(adj_nb)[0]]
                return result.tolist()
    
    return None


# ═══════════════════════════════════════════════
# Strategy 5: Row/Column Operations
# ═══════════════════════════════════════════════

def solve_row_col_ops(task):
    """Tasks where output rows/cols are permutations/selections of input rows/cols."""
    train = task['train']
    
    # Check: is each output row a copy of some input row?
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        if inp.shape[1] != out.shape[1]:
            return None
    
    # Try: output rows are sorted input rows
    sort_rules = [
        ('sort_rows_by_sum', lambda inp: np.array(sorted(range(inp.shape[0]), key=lambda r: np.sum(inp[r])))),
        ('sort_rows_by_nonzero', lambda inp: np.array(sorted(range(inp.shape[0]), key=lambda r: np.count_nonzero(inp[r])))),
        ('reverse_rows', lambda inp: np.arange(inp.shape[0]-1, -1, -1)),
        ('sort_cols_by_sum', lambda inp: None),  # handled separately
    ]
    
    for rule_name, rule_fn in sort_rules[:3]:
        all_match = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            if inp.shape != out.shape:
                all_match = False
                break
            perm = rule_fn(inp)
            if not np.array_equal(inp[perm], out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            perm = rule_fn(test_inp)
            return test_inp[perm].tolist()
    
    # Try column sorting
    for ex in train:
        inp, out = np.array(ex['input']), np.array(ex['output'])
        if inp.shape[0] != out.shape[0]:
            return None
    
    col_sorts = [
        ('sort_cols_by_sum', lambda inp: np.array(sorted(range(inp.shape[1]), key=lambda c: np.sum(inp[:, c])))),
        ('sort_cols_by_nonzero', lambda inp: np.array(sorted(range(inp.shape[1]), key=lambda c: np.count_nonzero(inp[:, c])))),
        ('reverse_cols', lambda inp: np.arange(inp.shape[1]-1, -1, -1)),
    ]
    
    for rule_name, rule_fn in col_sorts:
        all_match = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            if inp.shape != out.shape:
                all_match = False
                break
            perm = rule_fn(inp)
            if not np.array_equal(inp[:, perm], out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            perm = rule_fn(test_inp)
            return test_inp[:, perm].tolist()
    
    return None


# ═══════════════════════════════════════════════
# Strategy 6: Flood-fill based transforms
# ═══════════════════════════════════════════════

def solve_flood_fill(task):
    """Tasks involving flood-filling regions or propagating colors."""
    train = task['train']
    
    for ex in train:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None
    
    # Strategy: iteratively propagate non-bg cells to adjacent bg cells
    bg = find_bg(np.array(train[0]['input']))
    
    for max_iters in [1, 2, 5, 10, 30]:
        all_match = True
        for ex in train:
            inp, out = np.array(ex['input']), np.array(ex['output'])
            h, w = inp.shape
            
            current = inp.copy()
            for _ in range(max_iters):
                new = current.copy()
                changed = False
                for r in range(h):
                    for c in range(w):
                        if current[r, c] == bg:
                            # Check 4-neighbors
                            neighbors = []
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < h and 0 <= nc < w and current[nr, nc] != bg:
                                    neighbors.append(int(current[nr, nc]))
                            if neighbors:
                                # Fill with most common neighbor color
                                new[r, c] = Counter(neighbors).most_common(1)[0][0]
                                changed = True
                current = new
                if not changed:
                    break
            
            if not np.array_equal(current, out):
                all_match = False
                break
        
        if all_match:
            test_inp = np.array(task['test'][0]['input'])
            h, w = test_inp.shape
            current = test_inp.copy()
            for _ in range(max_iters):
                new = current.copy()
                changed = False
                for r in range(h):
                    for c in range(w):
                        if current[r, c] == bg:
                            neighbors = []
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < h and 0 <= nc < w and current[nr, nc] != bg:
                                    neighbors.append(int(current[nr, nc]))
                            if neighbors:
                                new[r, c] = Counter(neighbors).most_common(1)[0][0]
                                changed = True
                current = new
                if not changed:
                    break
            return current.tolist()
    
    return None


# ═══════════════════════════════════════════════
# Strategy 7: Color Frequency / Sorting
# ═══════════════════════════════════════════════

def solve_most_common_color(task):
    """Output is the most/least common color (various encodings)."""
    train = task['train']
    test_inp = np.array(task['test'][0]['input'])
    out0 = np.array(train[0]['output'])
    bg = find_bg(np.array(train[0]['input']))
    
    # Output is 1x1: the most/least common non-bg color
    if out0.size == 1:
        rules = [
            ('most_common_nonbg', lambda g: Counter(g[g != find_bg(g)].flatten().tolist()).most_common(1)[0][0]),
            ('least_common_nonbg', lambda g: Counter(g[g != find_bg(g)].flatten().tolist()).most_common()[-1][0]),
            ('most_common', lambda g: Counter(g.flatten().tolist()).most_common(1)[0][0]),
            ('n_unique_nonbg', lambda g: len(set(g.flatten().tolist()) - {find_bg(g)})),
        ]
        
        for rule_name, rule_fn in rules:
            all_match = True
            for ex in train:
                inp = np.array(ex['input'])
                out = np.array(ex['output']).flatten()[0]
                try:
                    pred = rule_fn(inp)
                    if pred != out:
                        all_match = False
                        break
                except Exception:
                    all_match = False
                    break
            
            if all_match:
                return [[rule_fn(test_inp)]]
    
    return None


# ═══════════════════════════════════════════════
# Strategy 8: Crop to non-bg bounding box
# ═══════════════════════════════════════════════

def solve_crop_variations(task):
    """Various cropping strategies."""
    train = task['train']
    test_inp = np.array(task['test'][0]['input'])
    
    bg = find_bg(test_inp)
    
    # Crop to bounding box of each non-bg color
    inp0 = np.array(train[0]['input'])
    out0 = np.array(train[0]['output'])
    
    colors = set(inp0.flatten().tolist()) - {bg}
    
    for color in colors:
        rows, cols = np.where(inp0 == color)
        if len(rows) == 0:
            continue
        crop = inp0[rows.min():rows.max()+1, cols.min():cols.max()+1]
        if grids_match(crop, out0):
            # Verify
            all_match = True
            for ex in train[1:]:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                r, c = np.where(inp == color)
                if len(r) == 0:
                    all_match = False
                    break
                cr = inp[r.min():r.max()+1, c.min():c.max()+1]
                if not grids_match(cr, out):
                    all_match = False
                    break
            
            if all_match:
                r, c = np.where(test_inp == color)
                if len(r) > 0:
                    return test_inp[r.min():r.max()+1, c.min():c.max()+1].tolist()
    
    # Crop to bounding box of all non-bg
    rows, cols = np.where(inp0 != bg)
    if len(rows) > 0:
        crop = inp0[rows.min():rows.max()+1, cols.min():cols.max()+1]
        if grids_match(crop, out0):
            all_match = True
            for ex in train[1:]:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                r, c = np.where(inp != find_bg(inp))
                if len(r) == 0:
                    all_match = False
                    break
                cr = inp[r.min():r.max()+1, c.min():c.max()+1]
                if not grids_match(cr, out):
                    all_match = False
                    break
            
            if all_match:
                r, c = np.where(test_inp != bg)
                if len(r) > 0:
                    return test_inp[r.min():r.max()+1, c.min():c.max()+1].tolist()
    
    return None


# ═══════════════════════════════════════════════
# Main 
# ═══════════════════════════════════════════════

def solve_task(task):
    """Try all strategies in order."""
    strategies = [
        ('neighborhood_r1', lambda t: solve_neighborhood_lut(t, radius=1)),
        ('abstract_features', solve_abstract_features),
        ('conditional_color', solve_conditional_color),
        ('grid_partition', solve_grid_partition),
        ('row_col_ops', solve_row_col_ops),
        ('flood_fill', solve_flood_fill),
        ('color_freq', solve_most_common_color),
        ('crop', solve_crop_variations),
        ('neighborhood_r2', lambda t: solve_neighborhood_lut(t, radius=2)),
    ]
    
    for name, fn in strategies:
        try:
            result = fn(task)
            if result is not None:
                return result, name
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
        # Predict without verification
        result_queue.put((False, f'{method}:wrong'))
        return
    
    result_queue.put((False, 'none'))


def solve_with_timeout(task_json, timeout_s=30):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(task_json, q))
    p.start()
    p.join(timeout=timeout_s)
    
    if p.is_alive():
        p.terminate()
        p.join(timeout=3)
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
    
    prev_solved = set()
    for f in ['arc_agi1_eval_results.json', 'arc_emerged_eval_results.json']:
        if os.path.exists(f):
            d = json.load(open(f))
            prev_solved |= set(d.get('solved_ids', []))
    
    log.info("=" * 60)
    log.info("🧠 ADVANCED ARC SOLVER — EVAL")
    log.info(f"Previously solved: {len(prev_solved)}/400")
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
        
        with open(os.path.join(eval_dir, fn)) as f:
            task = json.load(f)
        
        ok, method = solve_with_timeout(json.dumps(task), timeout_s=30)
        
        if ok:
            solved.append(tid)
            methods[method] = methods.get(method, 0) + 1
            tag = "🆕" if tid not in prev_solved else ""
            log.info(f"  ✅ {tid} [{method}] {tag}")
        
        if total % 50 == 0:
            wall = time.time() - t0
            new = len([t for t in solved if t not in prev_solved])
            log.info(f"  [{total}/400] {len(solved)} solved (+{new} new) | {wall:.0f}s")
        sys.stdout.flush()
    
    wall = time.time() - t0
    new_ids = [t for t in solved if t not in prev_solved]
    combined = prev_solved | set(solved)
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 RESULTS")
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
        'time_s': round(wall, 1),
    }
    with open('arc_advanced_eval_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    log.info(f"  → Saved arc_advanced_eval_results.json")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
