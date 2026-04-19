from itertools import combinations
from collections import defaultdict
import math

def transform(grid):
    import numpy as np
    inp = np.array(grid)
    H, W = inp.shape
    bg = int(np.bincount(inp.flatten()).argmax())
    
    non_bg = [(int(r), int(c), int(inp[r,c])) for r in range(H) for c in range(W) if inp[r,c] != bg]
    
    # Find rectangle corners (largest rectangle from any value with 4+ positions)
    best_rect = None
    for val in set(v for _,_,v in non_bg):
        positions = [(r,c) for r,c,v in non_bg if v == val]
        if len(positions) < 4:
            continue
        rows_all = sorted(set(r for r,c in positions))
        cols_all = sorted(set(c for r,c in positions))
        for r1, r2 in combinations(rows_all, 2):
            for c1, c2 in combinations(cols_all, 2):
                if all((r,c) in positions for r in [r1,r2] for c in [c1,c2]):
                    area = (r2-r1)*(c2-c1)
                    if best_rect is None or area > best_rect[4]:
                        best_rect = (r1, c1, r2, c2, area, val)
    
    r1, c1, r2, c2, _, corner_val = best_rect
    oh = r2 - r1 + 1
    ow = c2 - c1 + 1
    
    corner_set = {(r1,c1),(r1,c2),(r2,c1),(r2,c2)}
    corner_positions = {(0,0),(0,ow-1),(oh-1,0),(oh-1,ow-1)}
    
    scattered = [(r,c,v) for r,c,v in non_bg if (r,c) not in corner_set]
    
    groups = defaultdict(list)
    for r,c,v in scattered:
        groups[v].append((r,c))
    
    # For each value, find valid offsets
    valid_offsets = {}
    for v, pts in groups.items():
        offsets = []
        for a in range(oh):
            for b in range(ow):
                all_ok = True
                for r,c in pts:
                    or_ = (r+a) % oh
                    oc = (c+b) % ow
                    is_border = or_==0 or or_==oh-1 or oc==0 or oc==ow-1
                    is_corner = (or_, oc) in corner_positions
                    if not is_border or is_corner:
                        all_ok = False
                        break
                if all_ok:
                    offsets.append((a,b))
        valid_offsets[v] = offsets
    
    # Enumerate all conflict-free solutions
    values = sorted(valid_offsets.keys(), key=lambda v: len(valid_offsets[v]))
    all_solutions = []
    
    def get_cells(v, a, b):
        cells = set()
        for r,c in groups[v]:
            cells.add(((r+a) % oh, (c+b) % ow))
        return cells
    
    def enumerate_solutions(idx, occupied, current):
        if idx == len(values):
            all_solutions.append(dict(current))
            return
        v = values[idx]
        for a, b in valid_offsets[v]:
            cells = get_cells(v, a, b)
            # Check no conflict with OTHER values' cells
            if cells & occupied:
                continue
            current[v] = (a, b)
            enumerate_solutions(idx + 1, occupied | cells, current)
            del current[v]
        if len(all_solutions) > 100:  # safety limit
            return
    
    enumerate_solutions(0, set(), {})
    
    if len(all_solutions) == 0:
        # Fallback: try without conflict checking
        solution = {}
        for v in values:
            if valid_offsets[v]:
                solution[v] = valid_offsets[v][0]
    elif len(all_solutions) == 1:
        solution = all_solutions[0]
    else:
        # Pick solution with minimum max distance
        best_sol = None
        best_score = float('inf')
        for sol in all_solutions:
            max_dist = 0
            for v, (a, b) in sol.items():
                for r, c in groups[v]:
                    or_ = (r+a) % oh
                    oc = (c+b) % ow
                    ir = r1 + or_
                    ic = c1 + oc
                    dist = abs(r - ir) + abs(c - ic)
                    max_dist = max(max_dist, dist)
            if max_dist < best_score:
                best_score = max_dist
                best_sol = sol
        solution = best_sol
    
    # Build output
    out = [[bg] * ow for _ in range(oh)]
    out[0][0] = corner_val
    out[0][ow-1] = corner_val
    out[oh-1][0] = corner_val
    out[oh-1][ow-1] = corner_val
    
    for v, (a, b) in solution.items():
        for r,c in groups[v]:
            or_ = (r+a) % oh
            oc = (c+b) % ow
            out[or_][oc] = v
    
    return out
