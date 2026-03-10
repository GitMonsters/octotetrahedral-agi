"""Solver for 247ef758 — Place shapes at marker intersections"""
import json
from typing import List
from collections import Counter

def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    
    # Find divider column (all same non-zero value)
    divider_col = None
    for c in range(cols):
        col_vals = set(grid[r][c] for r in range(rows))
        if len(col_vals) == 1 and 0 not in col_vals:
            divider_col = c
            break
    
    rect_left = divider_col + 1
    rect_right = cols - 1
    
    # Border values
    left_border = [(r, grid[r][rect_left]) for r in range(rows)]
    top_border = [(c, grid[0][c]) for c in range(rect_left, rect_right + 1)]
    
    # Default border color
    border_vals = [v for _, v in left_border + top_border]
    default_border = Counter(border_vals).most_common(1)[0][0]
    
    # Markers on borders
    left_markers = {}
    for r, v in left_border:
        if v != default_border:
            left_markers.setdefault(v, []).append(r)
    
    top_markers = {}
    for c, v in top_border:
        if v != default_border:
            top_markers.setdefault(v, []).append(c)
    
    # Extract shapes from left side (cols 0 to divider_col-1)
    shapes = {}
    for r in range(rows):
        for c in range(divider_col):
            v = grid[r][c]
            if v != 0:
                shapes.setdefault(v, []).append((r, c))
    
    # Process shapes: sort by descending min row on left side so top shapes overwrite bottom
    shape_items = sorted(shapes.items(),
                        key=lambda x: min(r for r, c in x[1]), reverse=True)
    for color, cells in shape_items:
        if color in left_markers and color in top_markers:
            # Remove shape from left side
            for r, c in cells:
                output[r][c] = 0
            
            # Compute center of shape
            min_r = min(r for r, c in cells)
            max_r = max(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_c = max(c for r, c in cells)
            center_r = (min_r + max_r) / 2
            center_c = (min_c + max_c) / 2
            
            # Relative positions
            rel_cells = [(r - center_r, c - center_c) for r, c in cells]
            
            # Place at each marker intersection
            for mr in left_markers[color]:
                for mc in top_markers[color]:
                    for dr, dc in rel_cells:
                        nr = int(round(mr + dr))
                        nc = int(round(mc + dc))
                        if 0 <= nr < rows and 0 <= nc < cols:
                            output[nr][nc] = color
    
    return output


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        print(f"Train {i}: {'PASS ✓' if solve(ex['input'])==ex['output'] else 'FAIL'}")
