from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    flat = [c for row in grid for c in row]
    cc = Counter(flat)
    bg = cc.most_common(1)[0][0]
    
    # Find most common non-bg color
    non_bg_colors = [(c, cnt) for c, cnt in cc.items() if c != bg]
    if not non_bg_colors:
        return [[bg]*3 for _ in range(3)]
    
    main_color = max(non_bg_colors, key=lambda x: x[1])[0]
    
    # Count connected components of main color (4-connected)
    main_cells = set((r,c) for r in range(H) for c in range(W) if grid[r][c] == main_color)
    visited = set()
    N = 0
    for r,c in sorted(main_cells):
        if (r,c) in visited:
            continue
        N += 1
        stack = [(r,c)]
        while stack:
            cr, cc_ = stack.pop()
            if (cr, cc_) in visited or (cr, cc_) not in main_cells:
                continue
            visited.add((cr, cc_))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc_+dc))
    
    # The answer color is always 2
    answer = 2
    
    # Fixed position pattern based on count N
    # Positions are filled in this order:
    # (2,2), (0,2), (1,1), (2,0), (0,0), (1,2), (2,1), (0,1), (1,0)
    fill_order = [(2,2),(0,2),(1,1),(2,0),(0,0),(1,2),(2,1),(0,1),(1,0)]
    
    result = [[bg]*3 for _ in range(3)]
    for idx in range(min(N, 5)):
        r, c = fill_order[idx]
        result[r][c] = answer
    
    return result
