from collections import Counter, deque


def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    flat = [v for row in grid for v in row]
    color_counts = Counter(flat)
    bg = color_counts.most_common(1)[0][0]
    
    # Find non-bg cells
    non_bg_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
    
    if not non_bg_cells:
        return [row[:] for row in grid]
    
    non_bg_color = grid[non_bg_cells[0][0]][non_bg_cells[0][1]]
    
    # Find connected components using BFS
    non_bg_set = set(non_bg_cells)
    visited = set()
    components = []
    
    for r, c in non_bg_cells:
        if (r, c) in visited:
            continue
        comp = []
        queue = deque([(r, c)])
        visited.add((r, c))
        while queue:
            cr, cc = queue.popleft()
            comp.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if (nr, nc) in non_bg_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        components.append(comp)
    
    # Main block = largest connected component
    main_block = max(components, key=len)
    main_set = set(main_block)
    
    # Bounding box of main block
    top = min(r for r, c in main_block)
    bot = max(r for r, c in main_block)
    left = min(c for r, c in main_block)
    right = max(c for r, c in main_block)
    
    # Noise cells = non-bg cells not in main block
    noise = [(r, c) for r, c in non_bg_cells if (r, c) not in main_set]
    
    # Project each noise cell onto expanded boundary
    projections = set()
    for r, c in noise:
        pr = r
        pc = c
        if r < top:
            pr = top - 1
        elif r > bot:
            pr = bot + 1
        if c < left:
            pc = left - 1
        elif c > right:
            pc = right + 1
        if 0 <= pr < H and 0 <= pc < W:
            projections.add((pr, pc))
    
    # Build output
    out = [[bg] * W for _ in range(H)]
    
    # Paint main block
    for r, c in main_block:
        out[r][c] = non_bg_color
    
    # Paint projections
    for r, c in projections:
        out[r][c] = non_bg_color
    
    return out
