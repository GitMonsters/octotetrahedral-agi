def transform(grid):
    grid = [list(row) for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    # Find foreground color (non-background)
    fg = None
    for v in all_vals:
        if v != bg:
            fg = v
            break
    if fg is None:
        return grid
    
    # Find all foreground cells
    fg_cells = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == fg:
                fg_cells.add((r, c))
    
    # Find connected components
    def get_component(start, cells):
        visited = set()
        stack = [start]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or (r, c) not in cells:
                continue
            visited.add((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) in cells:
                    stack.append((nr, nc))
        return visited
    
    remaining = fg_cells.copy()
    components = []
    while remaining:
        start = next(iter(remaining))
        comp = get_component(start, remaining)
        components.append(comp)
        remaining -= comp
    
    # Find biggest component (the main rectangle)
    components.sort(key=len, reverse=True)
    big_rect = components[0]
    small_blocks = components[1:]
    
    # Get bounding box of big rectangle
    big_rows = [r for r, c in big_rect]
    big_cols = [c for r, c in big_rect]
    big_r1, big_r2 = min(big_rows), max(big_rows)
    big_c1, big_c2 = min(big_cols), max(big_cols)
    
    result = [row[:] for row in grid]
    
    # For each small block, draw lines extending from it in all directions
    for block in small_blocks:
        block_rows = [r for r, c in block]
        block_cols = [c for r, c in block]
        br1, br2 = min(block_rows), max(block_rows)
        bc1, bc2 = min(block_cols), max(block_cols)
        
        # Extend vertical lines from the block's columns
        for c in range(bc1, bc2+1):
            # Go upward from block
            for r in range(br1-1, -1, -1):
                if result[r][c] == bg:
                    result[r][c] = 3
            # Go downward from block  
            for r in range(br2+1, h):
                if result[r][c] == bg:
                    result[r][c] = 3
        
        # Extend horizontal lines from the block's rows
        for r in range(br1, br2+1):
            # Go left from block
            for c in range(bc1-1, -1, -1):
                if result[r][c] == bg:
                    result[r][c] = 3
            # Go right from block
            for c in range(bc2+1, w):
                if result[r][c] == bg:
                    result[r][c] = 3
    
    return result
