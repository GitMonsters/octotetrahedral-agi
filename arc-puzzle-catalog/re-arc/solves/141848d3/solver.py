def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    grid = [row[:] for row in input_grid]
    
    def is_uniform_row(r):
        return len(set(grid[r])) == 1
    
    def is_uniform_col(c):
        return len(set(grid[r][c] for r in range(rows))) == 1
    
    non_uniform_rows = [r for r in range(rows) if not is_uniform_row(r)]
    
    if len(non_uniform_rows) == 1:
        key_idx = non_uniform_rows[0]
        key = grid[key_idx]
        
        if key_idx >= rows // 2:  # key at bottom
            blank_val = grid[0][0]
            blank_rows = sorted([r for r in range(key_idx) if is_uniform_row(r) and grid[r][0] == blank_val], reverse=True)
            for i, r in enumerate(blank_rows):
                grid[r] = [key[i % len(key)]] * cols
        else:  # key at top
            blank_val = grid[rows-1][0]
            blank_rows = sorted([r for r in range(key_idx + 1, rows) if is_uniform_row(r) and grid[r][0] == blank_val])
            key_rev = key[::-1]
            for i, r in enumerate(blank_rows):
                grid[r] = [key_rev[i % len(key_rev)]] * cols
        return grid
    
    non_uniform_cols = [c for c in range(cols) if not is_uniform_col(c)]
    
    if len(non_uniform_cols) == 1:
        key_idx = non_uniform_cols[0]
        key = [grid[r][key_idx] for r in range(rows)]
        
        if key_idx >= cols // 2:  # key at right
            blank_val = grid[0][0]
            blank_cols = sorted([c for c in range(key_idx) if is_uniform_col(c) and grid[0][c] == blank_val], reverse=True)
            for i, c in enumerate(blank_cols):
                val = key[i % len(key)]
                for r in range(rows):
                    grid[r][c] = val
        else:  # key at left
            blank_val = grid[0][cols-1]
            blank_cols = sorted([c for c in range(key_idx + 1, cols) if is_uniform_col(c) and grid[0][c] == blank_val])
            key_rev = key[::-1]
            for i, c in enumerate(blank_cols):
                val = key_rev[i % len(key_rev)]
                for r in range(rows):
                    grid[r][c] = val
        return grid
    
    return grid
