"""
ARC-AGI Puzzle 69d93cb3 Solver

Rule: The grid is divided into cells by separator lines (rows/columns with
uniform values). For each local position within cells, if two or more cells
share markers of the same color at that position (same cell row or cell column),
fill all intermediate cells at that position with that color.
"""

from collections import Counter


def transform(grid):
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    
    # Find separator rows (rows where all values are the same)
    sep_rows = []
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep_rows.append(r)
    
    # Find separator cols (cols where all values are the same)
    sep_cols = []
    for c in range(cols):
        if len(set(grid[r][c] for r in range(rows))) == 1:
            sep_cols.append(c)
    
    # Build cell grid - list of (r1, r2, c1, c2) boundaries
    row_bounds = [-1] + sep_rows + [rows]
    col_bounds = [-1] + sep_cols + [cols]
    
    cell_grid = []
    for i in range(len(row_bounds) - 1):
        cell_row = []
        for j in range(len(col_bounds) - 1):
            r1, r2 = row_bounds[i] + 1, row_bounds[i + 1]
            c1, c2 = col_bounds[j] + 1, col_bounds[j + 1]
            if r1 < r2 and c1 < c2:
                cell_row.append((r1, r2, c1, c2))
        if cell_row:
            cell_grid.append(cell_row)
    
    if not cell_grid or not cell_grid[0]:
        return result
    
    # Get cell dimensions from first cell
    cr1, cr2, cc1, cc2 = cell_grid[0][0]
    cell_h, cell_w = cr2 - cr1, cc2 - cc1
    
    # Find background color (most common in cell interiors)
    cell_colors = Counter()
    for cell_row in cell_grid:
        for r1, r2, c1, c2 in cell_row:
            for r in range(r1, r2):
                for c in range(c1, c2):
                    cell_colors[grid[r][c]] += 1
    bg = cell_colors.most_common(1)[0][0]
    
    num_cell_rows = len(cell_grid)
    num_cell_cols = len(cell_grid[0])
    
    # For each local position within cells
    for local_r in range(cell_h):
        for local_c in range(cell_w):
            # Find all markers at this position, grouped by color
            markers = {}
            for ci in range(num_cell_rows):
                for cj in range(num_cell_cols):
                    r1, _, c1, _ = cell_grid[ci][cj]
                    val = grid[r1 + local_r][c1 + local_c]
                    if val != bg:
                        if val not in markers:
                            markers[val] = []
                        markers[val].append((ci, cj))
            
            # For each color with 2+ markers, draw lines between them
            for color, positions in markers.items():
                if len(positions) < 2:
                    continue
                
                # Group by cell row and cell column
                by_row = {}
                by_col = {}
                for ci, cj in positions:
                    by_row.setdefault(ci, []).append(cj)
                    by_col.setdefault(cj, []).append(ci)
                
                # Fill horizontal lines (same cell row)
                for ci, cols_list in by_row.items():
                    if len(cols_list) >= 2:
                        for cj in range(min(cols_list), max(cols_list) + 1):
                            r1, _, c1, _ = cell_grid[ci][cj]
                            result[r1 + local_r][c1 + local_c] = color
                
                # Fill vertical lines (same cell column)
                for cj, rows_list in by_col.items():
                    if len(rows_list) >= 2:
                        for ci in range(min(rows_list), max(rows_list) + 1):
                            r1, _, c1, _ = cell_grid[ci][cj]
                            result[r1 + local_r][c1 + local_c] = color
    
    return result
