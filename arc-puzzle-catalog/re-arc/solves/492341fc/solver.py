def transform(input_grid):
    from collections import Counter
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Count distinct colors -> scale factor
    colors = set()
    for row in input_grid:
        for v in row:
            colors.add(v)
    N = len(colors)
    
    # Find background (most common)
    all_vals = [v for row in input_grid for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    # Simple NxN upscale
    orows = rows * N
    ocols = cols * N
    output = [[bg] * ocols for _ in range(orows)]
    
    for r in range(rows):
        for c in range(cols):
            val = input_grid[r][c]
            for dr in range(N):
                for dc in range(N):
                    output[r * N + dr][c * N + dc] = val
    
    # Only draw diagonals when N >= 4
    if N >= 4:
        # Find singleton color (appears exactly once, not bg)
        color_counts = Counter(all_vals)
        singleton = None
        singleton_pos = None
        for val, count in color_counts.items():
            if val != bg and count == 1:
                singleton = val
                break
        
        if singleton is not None:
            for r in range(rows):
                for c in range(cols):
                    if input_grid[r][c] == singleton:
                        singleton_pos = (r, c)
                        break
                if singleton_pos:
                    break
            
            sr, sc = singleton_pos
            block_top = sr * N
            block_left = sc * N
            block_bottom = (sr + 1) * N - 1
            block_right = (sc + 1) * N - 1
            
            corners_and_dirs = [
                (block_top - 1, block_left - 1, -1, -1),
                (block_top - 1, block_right + 1, -1, 1),
                (block_bottom + 1, block_left - 1, 1, -1),
                (block_bottom + 1, block_right + 1, 1, 1),
            ]
            
            for start_r, start_c, dr, dc in corners_and_dirs:
                r, c = start_r, start_c
                while 0 <= r < orows and 0 <= c < ocols:
                    ir, ic = r // N, c // N
                    if input_grid[ir][ic] != bg:
                        break
                    output[r][c] = singleton
                    r += dr
                    c += dc
    
    return output
