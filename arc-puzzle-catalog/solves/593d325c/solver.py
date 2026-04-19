def transform(grid):
    H = len(grid)
    W = len(grid[0])
    S = min(H, W)
    
    # Scale each cell to SxS block
    out = [[0] * (W * S) for _ in range(H * S)]
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            for dr in range(S):
                for dc in range(S):
                    out[r * S + dr][c * S + dc] = color
    
    # Determine diagonal marker color
    # In this task the marker is always the color not in input
    # that appears in training outputs - consistently 1 for this instance
    in_colors = set()
    for row in grid:
        for v in row:
            in_colors.add(v)
    # Use 1 if not in input; otherwise find unused color
    if 1 not in in_colors:
        diag_color = 1
    else:
        diag_color = 0
        while diag_color in in_colors:
            diag_color += 1
    
    # Forward diagonal (\) on main diagonal interior cells
    for i in range(1, S - 1):
        a, b, c = grid[i-1][i-1], grid[i][i], grid[i+1][i+1]
        if a != b and b != c and a != c:
            for d in range(S):
                out[i * S + d][i * S + d] = diag_color
    
    # Backward diagonal (/) on main anti-diagonal interior cells
    for i in range(1, S - 1):
        ac = W - 1 - i
        ac_prev = W - 1 - (i - 1)
        ac_next = W - 1 - (i + 1)
        if 0 <= ac_next and ac_prev < W:
            a = grid[i-1][ac_prev]
            b = grid[i][ac]
            c = grid[i+1][ac_next]
            if a != b and b != c and a != c:
                for d in range(S):
                    out[i * S + d][ac * S + (S - 1 - d)] = diag_color
    
    return out
