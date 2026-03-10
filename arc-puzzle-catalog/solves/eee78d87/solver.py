"""Solver for eee78d87 — 3x3 Pattern Tiling with Maroon Center
6x6 input → 16x16 tiled output. Non-bg cells define a 3x3 tile (mod 3).
The tile is tiled across the 16x16 output, offset by the pattern center.
Center 6x6 region has black→maroon replacement."""
import json
from collections import Counter
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    tile = [[bg]*3 for _ in range(3)]
    non_bg = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != bg:
                tile[r%3][c%3] = 0
                non_bg.append((r, c))
    cr = ((min(r for r,_ in non_bg) + max(r for r,_ in non_bg)) // 2) % 3
    cc = ((min(c for _,c in non_bg) + max(c for _,c in non_bg)) // 2) % 3
    out = [[0]*16 for _ in range(16)]
    for r in range(16):
        for c in range(16):
            v = tile[(r+cr)%3][(c+cc)%3]
            if 5<=r<=10 and 5<=c<=10 and v==0: v = 9
            out[r][c] = v
    return out

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        print(f"Train {i}: {'PASS ✓' if solve(ex['input'])==ex['output'] else 'FAIL'}")
    for i, ex in enumerate(task['test']):
        print(f"Test {i}: {json.dumps(solve(ex['input']))}")
