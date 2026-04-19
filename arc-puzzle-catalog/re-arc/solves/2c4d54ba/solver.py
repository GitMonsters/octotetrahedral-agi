def transform(grid):
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background (most common color)
    cc = Counter()
    for r in range(rows):
        for c in range(cols):
            cc[grid[r][c]] += 1
    bg = cc.most_common(1)[0][0]
    
    # Find non-bg colors
    non_bg = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg.add(grid[r][c])
    
    if len(non_bg) <= 1:
        return [row[:] for row in grid]
    
    # Identify frame color: bbox is a solid non-bg rectangle containing another color
    frame_color = None
    inner_color = None
    for color in non_bg:
        cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        r0 = min(r for r, c in cells)
        r1 = max(r for r, c in cells)
        c0 = min(c for r, c in cells)
        c1 = max(c for r, c in cells)
        solid = all(grid[r][c] != bg for r in range(r0, r1+1) for c in range(c0, c1+1))
        has_other = any(grid[r][c] != color for r in range(r0, r1+1) for c in range(c0, c1+1))
        if solid and has_other:
            frame_color = color
            inner_color = (non_bg - {color}).pop()
            break
    
    if frame_color is None:
        return [row[:] for row in grid]
    
    # Find frame bounding box
    frame_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == frame_color]
    fr0 = min(r for r, c in frame_cells)
    fr1 = max(r for r, c in frame_cells)
    fc0 = min(c for r, c in frame_cells)
    fc1 = max(c for r, c in frame_cells)
    
    # Find inner block inside frame
    inner_in_frame = [(r, c) for r in range(fr0, fr1+1) for c in range(fc0, fc1+1) if grid[r][c] == inner_color]
    ir0 = min(r for r, c in inner_in_frame)
    ir1 = max(r for r, c in inner_in_frame)
    ic0 = min(c for r, c in inner_in_frame)
    ic1 = max(c for r, c in inner_in_frame)
    
    # Compute intrinsic margins
    tm = ir0 - fr0
    bm = fr1 - ir1
    lm = ic0 - fc0
    rm = fc1 - ic1
    vm = max(tm, bm)
    hm = max(lm, rm)
    
    # Find all inner blocks via connected components
    all_inner = set((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == inner_color)
    remaining = set(all_inner)
    blocks = []
    while remaining:
        start = min(remaining)
        queue = [start]
        comp = {start}
        while queue:
            cr, cc_val = queue.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc_val+dc
                if (nr, nc) in remaining and (nr, nc) not in comp:
                    comp.add((nr, nc))
                    queue.append((nr, nc))
        for cell in comp:
            remaining.discard(cell)
        br0 = min(r for r, c in comp)
        br1 = max(r for r, c in comp)
        bc0 = min(c for r, c in comp)
        bc1 = max(c for r, c in comp)
        blocks.append((br0, bc0, br1, bc1))
    
    blocks.sort(key=lambda b: (b[0], b[1]))
    
    # Find which block is currently framed
    current_idx = None
    for idx, (br0, bc0, br1, bc1) in enumerate(blocks):
        if br0 >= fr0 and br1 <= fr1 and bc0 >= fc0 and bc1 <= fc1:
            current_idx = idx
            break
    
    # Move frame to previous block (index - 1)
    if current_idx is None:
        return [row[:] for row in grid]
    new_idx = current_idx - 1
    if new_idx < 0:
        new_idx = len(blocks) - 1
    
    nb_r0, nb_c0, nb_r1, nb_c1 = blocks[new_idx]
    
    # Compute new frame position (clipped to grid)
    nf_r0 = max(0, nb_r0 - vm)
    nf_r1 = min(rows - 1, nb_r1 + vm)
    nf_c0 = max(0, nb_c0 - hm)
    nf_c1 = min(cols - 1, nb_c1 + hm)
    
    # Build output
    output = [[bg] * cols for _ in range(rows)]
    
    # Place all inner blocks
    for br0, bc0, br1, bc1 in blocks:
        for r in range(br0, br1+1):
            for c in range(bc0, bc1+1):
                output[r][c] = inner_color
    
    # Place new frame (overwriting background and inner if needed)
    for r in range(nf_r0, nf_r1+1):
        for c in range(nf_c0, nf_c1+1):
            output[r][c] = frame_color
    
    # Re-place the inner block INSIDE the frame (on top of frame)
    for r in range(nb_r0, nb_r1+1):
        for c in range(nb_c0, nb_c1+1):
            output[r][c] = inner_color
    
    return output
