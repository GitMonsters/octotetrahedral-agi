from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find blocks by detecting color transitions in the first row
    # Each block has a dominant fill color; separator columns have a different color
    # Strategy: find contiguous column ranges that form blocks
    
    # A block boundary is where the "pattern" changes
    # Use first and last row to identify block edges
    # A separator column has a uniform value that matches neither the left nor right block
    
    # Better strategy: find contiguous "block" regions
    # Group columns by their dominant (most common) value across all rows
    col_dominant = []
    for c in range(W):
        col = [grid[r][c] for r in range(H)]
        col_dominant.append(Counter(col).most_common(1)[0][0])
    
    # Find blocks: contiguous runs of columns with the same dominant color
    # But also handle separator columns (single columns between blocks)
    blocks = []
    c = 0
    while c < W:
        start = c
        dom = col_dominant[c]
        # Extend while columns have the same dominant
        while c < W and col_dominant[c] == dom:
            c += 1
        blocks.append((start, c, dom))
    
    # Now merge adjacent blocks that should be one:
    # A "separator" block is a narrow block (1-2 cols) with a color shared with its neighbors' secondary
    # Actually: each block is defined by its fill color
    # If a narrow block's color appears as the frame/border in adjacent blocks, merge
    
    # Simpler approach: identify "real" blocks vs separators
    # A separator has width <= 2 AND its color appears in the adjacent blocks
    real_blocks = []
    i = 0
    while i < len(blocks):
        start, end, dom = blocks[i]
        width = end - start
        
        # Check if this block is a separator
        is_sep = False
        if width <= 2 and len(blocks) > 1:
            # Check if its color appears as minority in adjacent blocks
            left_has = False
            right_has = False
            if i > 0:
                ls, le, ld = blocks[i-1]
                left_vals = set(grid[r][c] for r in range(H) for c in range(ls, le))
                left_has = dom in left_vals
            if i < len(blocks) - 1:
                rs, re, rd = blocks[i+1]
                right_vals = set(grid[r][c] for r in range(H) for c in range(rs, re))
                right_has = dom in right_vals
            is_sep = left_has or right_has
        
        if not is_sep:
            real_blocks.append((start, end))
        i += 1
    
    if not real_blocks:
        real_blocks = [(s, e) for s, e, d in blocks]
    
    N = len(real_blocks)
    
    # For each real block, determine output value
    values = []
    for (c0, c1) in real_blocks:
        block_flat = [grid[r][c] for r in range(H) for c in range(c0, c1)]
        cnt = Counter(block_flat)
        majority = cnt.most_common(1)[0][0]
        colors = set(cnt.keys())
        
        if len(colors) == 1:
            # Solid block
            values.append(7)
        elif majority == bg:
            # Majority matches bg
            values.append(7)
        elif bg in colors:
            # Majority != bg, but block contains bg as minority
            values.append(1)
        else:
            # Majority != bg and no bg in block
            values.append(6)
    
    # Build N x N output
    return [[v] * N for v in values]
