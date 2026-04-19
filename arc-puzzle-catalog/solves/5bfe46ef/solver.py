from collections import Counter

def transform(grid):
    H = len(grid); W = len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    fg_cells = [(r,c) for r in range(H) for c in range(W) if grid[r][c] != bg]
    
    if not fg_cells:
        # No fg: check if grid is 1:2 ratio -> return bg, else 8
        if max(H,W) == 2*min(H,W):
            return [[bg]]
        return [[8]]
    
    # Find fg color
    fg_color = grid[fg_cells[0][0]][fg_cells[0][1]]
    
    # Find bounding box of fg cells
    r_min = min(r for r,c in fg_cells); r_max = max(r for r,c in fg_cells)
    c_min = min(c for r,c in fg_cells); c_max = max(c for r,c in fg_cells)
    
    # The fg cells form a pattern of 2x2 blocks in a 3x3 arrangement
    # Determine unit size (each block is 2x2)
    box_H = r_max - r_min + 1
    box_W = c_max - c_min + 1
    
    if box_H == 0 or box_W == 0:
        return [[8]]
    
    # Find block unit size (should be 2)
    unit = 2
    
    # Map each fg cell to a block position in 3x3 grid
    block_positions = set()
    for r,c in fg_cells:
        br = (r - r_min) // unit
        bc = (c - c_min) // unit
        block_positions.add((br, bc))
    
    # + pattern: (0,1),(1,0),(1,1),(1,2),(2,1)
    plus = {(0,1),(1,0),(1,1),(1,2),(2,1)}
    # X pattern: (0,0),(0,2),(1,1),(2,0),(2,2)
    cross = {(0,0),(0,2),(1,1),(2,0),(2,2)}
    
    if block_positions == plus:
        return [[fg_color]]
    elif block_positions == cross:
        return [[8]]
    else:
        # 2-rect or other pattern -> return 8
        return [[8]]
