import math
from collections import Counter
from functools import reduce

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    val_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            val_counts[grid[r][c]] += 1
    bg = val_counts.most_common(1)[0][0]
    mc = val_counts.most_common()
    glv_count = mc[1][0] if len(mc) > 1 else bg
    
    # Find special positions (not bg, not glv_count)
    sp_rows, sp_cols = set(), set()
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg and v != glv_count:
                sp_rows.add(r); sp_cols.add(c)
    sp_rows = sorted(sp_rows)
    sp_cols = sorted(sp_cols)
    
    row_nonbg = [sum(1 for c in range(cols) if grid[r][c] != bg) for r in range(rows)]
    col_nonbg = [sum(1 for r in range(rows) if grid[r][c] != bg) for c in range(cols)]
    
    def fill_and_extend(positions, max_val, counts):
        if len(positions) < 2:
            return positions
        diffs = [positions[i+1] - positions[i] for i in range(len(positions) - 1)]
        spacing = reduce(math.gcd, diffs)
        if spacing == 0:
            return positions
        filled = list(range(positions[0], positions[-1] + 1, spacing))
        # Extend backward
        pos = filled[0] - spacing
        while pos >= 0 and counts[pos] > 0:
            filled.insert(0, pos)
            pos -= spacing
        # Extend forward
        pos = filled[-1] + spacing
        while pos < max_val and counts[pos] > 0:
            filled.append(pos)
            pos += spacing
        return filled
    
    grid_rows = fill_and_extend(sp_rows, rows, row_nonbg)
    grid_cols = fill_and_extend(sp_cols, cols, col_nonbg)
    
    # Find structure-based glv
    glv = bg
    if len(grid_rows) >= 2 and len(grid_cols) >= 1:
        # Find a non-intersection position on a grid line
        gc = grid_cols[0]
        for r in range(rows):
            if r not in set(grid_rows):
                glv = grid[r][gc]
                break
    
    # Build output with modified rule
    result = []
    for ri in range(len(grid_rows) - 1):
        row = []
        for ci in range(len(grid_cols) - 1):
            r1, r2 = grid_rows[ri], grid_rows[ri+1]
            c1, c2 = grid_cols[ci], grid_cols[ci+1]
            corners = [grid[r1][c1], grid[r1][c2], grid[r2][c1], grid[r2][c2]]
            if len(set(corners)) == 1 and corners[0] != bg and corners[0] != glv:
                row.append(corners[0])
            else:
                row.append(bg)
        result.append(row)
    
    # Trim all-bg rows from bottom and cols from right
    while len(result) > 1 and all(v == bg for v in result[-1]):
        result.pop()
    while len(result) > 0 and len(result[0]) > 1 and all(result[r][-1] == bg for r in range(len(result))):
        for r in range(len(result)):
            result[r].pop()
    
    return result
