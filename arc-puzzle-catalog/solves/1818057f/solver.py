"""Solver for 1818057f — Plus Pattern Detection
Find plus/cross shapes (center + 4 orthogonal neighbors all yellow=4)
and replace them with azure (8)."""
import json
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if (grid[r][c] == 4 and grid[r-1][c] == 4 and grid[r+1][c] == 4
                and grid[r][c-1] == 4 and grid[r][c+1] == 4):
                out[r][c] = 8
                out[r-1][c] = 8
                out[r+1][c] = 8
                out[r][c-1] = 8
                out[r][c+1] = 8
    return out

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        print(f"Train {i}: {'PASS ✓' if solve(ex['input'])==ex['output'] else 'FAIL'}")
    for i, ex in enumerate(task['test']):
        print(f"Test {i}: predicted")
        pred = solve(ex['input'])
        for row in pred: print(row)
