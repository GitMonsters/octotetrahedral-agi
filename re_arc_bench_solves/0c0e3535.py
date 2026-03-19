"""
ARC Puzzle 0c0e3535 Solver

Pattern: Compare diagonally-opposite quadrant fills of the minority-color shape.
- If TL+BR > TR+BL: output [[8]]
- Otherwise: output [[2]]
"""

from collections import Counter

def transform(grid):
    """
    Identify the shape (minority color), compute quadrant fills,
    and return [[8]] if main diagonal dominates, else [[2]].
    """
    # Count colors to find background vs shape color
    color_counts = Counter(c for row in grid for c in row)
    sorted_colors = color_counts.most_common()
    
    # Background is most common, shape is second most common
    shape_color = sorted_colors[1][0]
    
    # Find bounding box of shape
    cells = []
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == shape_color:
                cells.append((r, c))
    
    min_r = min(r for r, c in cells)
    max_r = max(r for r, c in cells)
    min_c = min(c for r, c in cells)
    max_c = max(c for r, c in cells)
    
    # Normalize to bounding box coordinates
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    
    # Count cells in each quadrant
    mid_r = h / 2
    mid_c = w / 2
    
    tl = tr = bl = br = 0
    for r, c in cells:
        nr, nc = r - min_r, c - min_c  # normalized coords
        if nr < mid_r:
            if nc < mid_c:
                tl += 1
            else:
                tr += 1
        else:
            if nc < mid_c:
                bl += 1
            else:
                br += 1
    
    # Compare diagonals
    diag1 = tl + br  # main diagonal
    diag2 = tr + bl  # anti-diagonal
    
    if diag1 > diag2:
        return [[8]]
    else:
        return [[2]]
