"""
ARC Puzzle 0d631f30 Solver

Pattern: Find 4 corner markers forming a rectangle, extract the region inside,
and replace the shape color with the marker color.

- Markers: 4 pixels of the same color forming a rectangle's corners
- Shape: A pattern drawn in a third color inside the marker region
- Output: The region inside the markers, with shape color replaced by marker color
"""
from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Count colors
    color_counts = Counter()
    for row in grid:
        for val in row:
            color_counts[val] += 1
    
    # Background is most common
    bg_color = color_counts.most_common(1)[0][0]
    
    # Find marker positions (4 corners forming a rectangle)
    marker_color = None
    shape_color = None
    marker_positions = []
    
    for color in color_counts:
        if color == bg_color:
            continue
        positions = []
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val == color:
                    positions.append((r, c))
        
        if len(positions) == 4:
            r_vals = sorted(set(p[0] for p in positions))
            c_vals = sorted(set(p[1] for p in positions))
            if len(r_vals) == 2 and len(c_vals) == 2:
                corners = {(r_vals[0], c_vals[0]), (r_vals[0], c_vals[1]),
                           (r_vals[1], c_vals[0]), (r_vals[1], c_vals[1])}
                if set(positions) == corners:
                    marker_color = color
                    marker_positions = positions
        else:
            # Not 4 positions - likely the shape
            if shape_color is None:
                shape_color = color
    
    # If no markers found (edge case with all same color)
    if not marker_positions:
        return [[bg_color] * 9 for _ in range(9)]
    
    # Get bounding rectangle from markers
    r_vals = [p[0] for p in marker_positions]
    c_vals = [p[1] for p in marker_positions]
    min_r, max_r = min(r_vals), max(r_vals)
    min_c, max_c = min(c_vals), max(c_vals)
    
    # Extract region inside markers (exclusive of marker positions)
    output = []
    for r in range(min_r + 1, max_r):
        row = []
        for c in range(min_c + 1, max_c):
            val = grid[r][c]
            if shape_color is not None and val == shape_color:
                val = marker_color
            row.append(val)
        output.append(row)
    
    return output
