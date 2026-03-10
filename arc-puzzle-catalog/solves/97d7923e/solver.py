"""Solver for 97d7923e — Vertical Bar Selection
Vertical bars with colored endpoints and filler. A lone marker's height
selects which bar (from outermost) gets its filler replaced by endpoint color."""
import json
from typing import List
from collections import defaultdict

def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    
    bars = []
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        non_zero = [(r, col_vals[r]) for r in range(rows) if col_vals[r] != 0]
        for i in range(len(non_zero)):
            for j in range(i+1, len(non_zero)):
                r1, c1 = non_zero[i]
                r2, c2 = non_zero[j]
                if c1 != c2 or r2 - r1 < 2: continue
                filler = col_vals[r1+1]
                if filler == 0 or filler == c1: continue
                if all(col_vals[r] == filler for r in range(r1+1, r2)):
                    bars.append((c, r1, r2, c1, filler))
    
    bar_groups = defaultdict(list)
    bar_cells = set()
    for bar in bars:
        c, r1, r2, ec, fc = bar
        bar_groups[ec].append(bar)
        for r in range(r1, r2+1): bar_cells.add((r, c))
    for ec in bar_groups:
        bar_groups[ec].sort(key=lambda b: b[1])
    
    lone_markers = defaultdict(list)
    for color in bar_groups:
        color_cells = {(r,c) for r in range(rows) for c in range(cols) 
                       if grid[r][c] == color and (r,c) not in bar_cells}
        visited = set()
        for (r, c) in sorted(color_cells):
            if (r,c) in visited: continue
            h = 0
            rr = r
            while (rr, c) in color_cells:
                visited.add((rr, c)); h += 1; rr += 1
            lone_markers[color].append(h)
    
    for ec, group_bars in bar_groups.items():
        if ec not in lone_markers: continue
        idx = max(lone_markers[ec]) - 1
        if 0 <= idx < len(group_bars):
            c, r1, r2, _, _ = group_bars[idx]
            for r in range(r1+1, r2): out[r][c] = ec
    return out

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        print(f"Train {i}: {'PASS ✓' if solve(ex['input'])==ex['output'] else 'FAIL'}")
