"""Solver for 31f7f899 — Staircase Extent Sorting
Vertical colored lines cross a horizontal purple line. Sort line extents ascending left-to-right."""
import json
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    bg = 8  # azure background
    
    # Find center row: the one with the most purple(6) cells
    center_row = max(range(rows), key=lambda r: sum(1 for c in range(cols) if grid[r][c] == 6))
    
    # Find vertical lines: for each column, find color and extent above/below center
    lines = []
    for c in range(cols):
        color = grid[center_row][c]
        if color == 6 or color == bg:
            continue
        ext_above = 0
        for d in range(1, rows):
            if center_row - d >= 0 and grid[center_row - d][c] == color:
                ext_above += 1
            else:
                break
        ext_below = 0
        for d in range(1, rows):
            if center_row + d < rows and grid[center_row + d][c] == color:
                ext_below += 1
            else:
                break
        lines.append((c, color, ext_above, ext_below))
    
    # Sort extent pairs ascending by total extent
    pairs_sorted = sorted([(a, b) for _, _, a, b in lines], key=lambda p: p[0] + p[1])
    
    # Rebuild grid
    out = [[bg] * cols for _ in range(rows)]
    for c in range(cols):
        out[center_row][c] = 6
    
    # Place lines with new extents
    for i, (c, color, _, _) in enumerate(lines):
        new_above, new_below = pairs_sorted[i]
        out[center_row][c] = color
        for d in range(1, new_above + 1):
            if center_row - d >= 0:
                out[center_row - d][c] = color
        for d in range(1, new_below + 1):
            if center_row + d < rows:
                out[center_row + d][c] = color
    
    return out

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        match = result == ex['output']
        print(f"Train {i}: {'PASS ✓' if match else 'FAIL ✗'}")
        if not match:
            diffs = [(r,c,result[r][c],ex['output'][r][c]) for r in range(len(result)) for c in range(len(result[0])) if result[r][c]!=ex['output'][r][c]]
            for r,c,g,e in diffs[:10]: print(f"  ({r},{c}): got {g} expected {e}")
