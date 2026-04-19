def transform(grid):
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    
    counter = Counter()
    for r in grid:
        counter.update(r)
    bg = counter.most_common(1)[0][0]
    
    # Find non-bg colors
    non_bg_colors = {v for v in counter if v != bg}
    
    # Find block color: forms a perfect rectangle
    block_color = None
    block_top = block_bot = block_left = block_right = None
    
    for color in non_bg_colors:
        cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        if not cells:
            continue
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        expected = (max_r - min_r + 1) * (max_c - min_c + 1)
        if len(cells) == expected and expected > 1:
            block_color = color
            block_top, block_bot = min_r, max_r
            block_left, block_right = min_c, max_c
            break
    
    out = [[bg] * cols for _ in range(rows)]
    
    if block_color is None:
        return out
    
    # Keep the block
    for r in range(block_top, block_bot + 1):
        for c in range(block_left, block_right + 1):
            out[r][c] = block_color
    
    # Scatter = all non-bg, non-block cells
    scatter_color = None
    for color in non_bg_colors:
        if color != block_color:
            scatter_color = color
            break
    
    if scatter_color is None:
        return out
    
    # Check if block is horizontal (spans full width) or vertical (spans full height)
    if block_left == 0 and block_right == cols - 1:
        # Horizontal block: compress vertically
        for c in range(cols):
            # Count scatter above
            above_count = sum(1 for r in range(block_top) if grid[r][c] == scatter_color)
            # Place above_count cells descending from top of block
            for i in range(above_count):
                out[block_top - 1 - i][c] = block_color
            
            # Count scatter below
            below_count = sum(1 for r in range(block_bot + 1, rows) if grid[r][c] == scatter_color)
            for i in range(below_count):
                out[block_bot + 1 + i][c] = block_color
    
    elif block_top == 0 and block_bot == rows - 1:
        # Vertical block: compress horizontally
        for r in range(rows):
            left_count = sum(1 for c in range(block_left) if grid[r][c] == scatter_color)
            for i in range(left_count):
                out[r][block_left - 1 - i] = block_color
            
            right_count = sum(1 for c in range(block_right + 1, cols) if grid[r][c] == scatter_color)
            for i in range(right_count):
                out[r][block_right + 1 + i] = block_color
    else:
        # General case - try both directions
        for c in range(block_left, block_right + 1):
            above_count = sum(1 for r in range(block_top) if grid[r][c] == scatter_color)
            for i in range(above_count):
                out[block_top - 1 - i][c] = block_color
            below_count = sum(1 for r in range(block_bot + 1, rows) if grid[r][c] == scatter_color)
            for i in range(below_count):
                out[block_bot + 1 + i][c] = block_color
        
        for r in range(block_top, block_bot + 1):
            left_count = sum(1 for c in range(block_left) if grid[r][c] == scatter_color)
            for i in range(left_count):
                out[r][block_left - 1 - i] = block_color
            right_count = sum(1 for c in range(block_right + 1, cols) if grid[r][c] == scatter_color)
            for i in range(right_count):
                out[r][block_right + 1 + i] = block_color
    
    return out
