#!/usr/bin/env python3
"""db695cfb solver: Blue diagonal with purple perpendicular crossings.

Rule:
1. Find all blue(1) and purple(6) dots
2. Find pairs of blue dots that lie on exact 45° diagonals
3. For each pair, draw diagonal line (fill with blue)
4. Purple dots ON a blue diagonal create perpendicular diagonals to grid boundaries
5. Non-paired blues and non-path purples stay unchanged
"""
import json, sys
from collections import Counter

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    out = [r[:] for r in grid]
    
    blues = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    purples = set((r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 6)
    
    # Find pairs of blue dots on exact 45° diagonals
    used = set()
    pairs = []
    for i in range(len(blues)):
        for j in range(i+1, len(blues)):
            b1, b2 = blues[i], blues[j]
            dr = abs(b2[0] - b1[0])
            dc = abs(b2[1] - b1[1])
            if dr == dc and dr > 0:
                pairs.append((b1, b2))
                used.add(i)
                used.add(j)
    
    for b1, b2 in pairs:
        dr = 1 if b2[0] > b1[0] else -1
        dc = 1 if b2[1] > b1[1] else -1
        
        # Collect path cells
        path_cells = set()
        r, c = b1
        while True:
            path_cells.add((r, c))
            if (r, c) == b2:
                break
            r += dr
            c += dc
        
        # Fill blue on path (skip purples)
        for (r, c) in path_cells:
            if (r, c) not in purples:
                out[r][c] = 1
        
        # Perpendicular direction
        perp_dr, perp_dc = dr, -dc
        
        # For each purple on this path, draw perpendicular diagonal
        for pr, pc in purples:
            if (pr, pc) not in path_cells:
                continue
            for sign in [1, -1]:
                sr, sc = pr + sign * perp_dr, pc + sign * perp_dc
                while 0 <= sr < rows and 0 <= sc < cols:
                    if out[sr][sc] != 1:  # don't overwrite blue
                        out[sr][sc] = 6
                    sr += sign * perp_dr
                    sc += sign * perp_dc
    
    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        expected = ex['output']
        match = result == expected
        print(f"Train {i}: {'PASS ✓' if match else 'FAIL ✗'}")
        if not match:
            diffs = [(r,c,result[r][c],expected[r][c]) for r in range(len(expected)) for c in range(len(expected[0])) if result[r][c]!=expected[r][c]]
            print(f"  {len(diffs)} wrong: {diffs[:10]}")
