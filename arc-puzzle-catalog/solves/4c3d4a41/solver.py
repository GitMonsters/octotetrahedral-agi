import json, sys

def solve(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    
    # Find separator: all-0 column with non-0 content on both sides
    sep = None
    for c in range(1, C-1):
        if all(grid[r][c] == 0 for r in range(R)):
            has_left = any(grid[r][lc] != 0 for r in range(R) for lc in range(c))
            has_right = any(grid[r][rc] != 0 for r in range(R) for rc in range(c+1, C))
            if has_left and has_right:
                sep = c
                break
    if sep is None:
        return grid
    
    # Find all left ⬜(5) cells (staircase)
    left_white = set()
    for r in range(R):
        for c in range(sep):
            if grid[r][c] == 5:
                left_white.add((r, c))
    if not left_white:
        return grid
    
    # Find data columns in right half (cols with non-0, non-5 values)
    right_data_cols = []
    for c in range(sep+1, C):
        for r in range(R):
            if grid[r][c] not in (0, 5):
                right_data_cols.append(c)
                break
    
    # Find staircase data columns in left (cols with ⬜ above base row)
    base_row = max(r for r, c in left_white)
    left_data_cols = sorted(set(c for r, c in left_white if r < base_row))
    
    # Compute offset: first left data col -> first right data col
    if not left_data_cols or not right_data_cols:
        return grid
    offset = right_data_cols[0] - left_data_cols[0]
    
    # Clear left half
    for r in range(R):
        for c in range(sep):
            out[r][c] = 0
    
    # Map staircase positions to right half
    staircase_right = set()
    for r, c in left_white:
        tc = c + offset
        if sep < tc < C:
            staircase_right.add((r, tc))
    
    # Find right frame interior top row
    right_border = sep + 1
    int_top = None
    for r in range(R):
        if grid[r][right_border] == 5:
            if int_top is None:
                int_top = r + 1
    if int_top is None:
        int_top = 0
    
    # For each data column, compress colors into non-staircase rows
    for dc in right_data_cols:
        # Staircase rows for this column
        stairs = set(r for r in range(R) if (r, dc) in staircase_right)
        
        # Active area: from int_top to max staircase row
        active_top = int_top
        active_bot = base_row
        
        # Extract colors from original (non-0, non-5)
        colors = []
        for r in range(active_top, active_bot + 1):
            v = grid[r][dc]
            if v not in (0, 5):
                colors.append(v)
        
        # Available rows (active, not staircase)
        avail = [r for r in range(active_top, active_bot + 1) if r not in stairs]
        
        # Place colors at BOTTOM of available, ⬜ fills top
        n = min(len(colors), len(avail))
        for j, r in enumerate(avail):
            if j < len(avail) - n:
                out[r][dc] = 5
            else:
                color_idx = j - (len(avail) - n)
                out[r][dc] = colors[color_idx]
        
        # Staircase rows within active area → ⬜
        for r in stairs:
            if active_top <= r <= active_bot:
                out[r][dc] = 5
    
    # Apply staircase ⬜ to spacing columns in right half
    for r, c in left_white:
        tc = c + offset
        if sep < tc < C and tc not in right_data_cols:
            out[r][tc] = 5
    
    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        ok = result == ex['output']
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex['output'][r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, exp {ex['output'][r][c]}")
