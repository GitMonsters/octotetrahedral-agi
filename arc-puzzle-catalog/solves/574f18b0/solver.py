def transform(grid):
    H = len(grid)
    W = len(grid[0])
    from collections import Counter
    
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common()[0][0]
    
    # Find non-bg pixels
    pixels = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                pixels[(r, c)] = grid[r][c]
    
    if not pixels:
        return [row[:] for row in grid]
    
    # Find anchor at one of the 4 corner positions (1,1), (1,W-2), (H-2,1), (H-2,W-2)
    corner_positions = [(1, 1), (1, W-2), (H-2, 1), (H-2, W-2)]
    anchor = None
    for pos in corner_positions:
        if pos in pixels:
            anchor = pos
            break
    
    if anchor is None:
        return [row[:] for row in grid]
    
    ar, ac = anchor
    
    # Map pixels to symmetric matrix coordinates
    M = {}
    for (r, c), v in pixels.items():
        i = abs(r - ar) // 2
        j = abs(c - ac) // 2
        M[(i, j)] = v
    
    max_k = max(max(i, j) for i, j in M)
    
    # Generate output with concentric rectangular rings
    out = [[bg] * W for _ in range(H)]
    
    for k in range(max_k + 1):
        rk = 1 + 2 * k
        ck = 1 + 2 * k
        rrk = H - 2 - 2 * k
        rck = W - 2 - 2 * k
        
        # Draw corners from diagonal M[k][k]
        if (k, k) in M:
            color = M[(k, k)]
            for r in [rk, rrk]:
                for c in [ck, rck]:
                    if 0 <= r < H and 0 <= c < W:
                        out[r][c] = color
        
        # Draw edges from off-diagonal M[k][k+1]
        edge_color = M.get((k, k + 1))
        if edge_color is None:
            edge_color = M.get((k + 1, k))
        if edge_color is not None:
            nk = k + 1
            rnk = 1 + 2 * nk
            cnk = 1 + 2 * nk
            rrnk = H - 2 - 2 * nk
            rcnk = W - 2 - 2 * nk
            
            # Top and bottom edges of ring k, at columns of ring k+1
            for c in range(cnk, rcnk + 1, 2):
                if 0 <= c < W:
                    if 0 <= rk < H:
                        out[rk][c] = edge_color
                    if 0 <= rrk < H:
                        out[rrk][c] = edge_color
            
            # Left and right edges of ring k, at rows of ring k+1
            for r in range(rnk, rrnk + 1, 2):
                if 0 <= r < H:
                    if 0 <= ck < W:
                        out[r][ck] = edge_color
                    if 0 <= rck < W:
                        out[r][rck] = edge_color
    
    return out
