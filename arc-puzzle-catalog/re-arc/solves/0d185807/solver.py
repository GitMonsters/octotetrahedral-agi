def transform(input_grid):
    N, M = len(input_grid), len(input_grid[0])
    # Count cells with value 3 (the "background" of the fractal)
    K = sum(1 for r in input_grid for v in r if v == 3)
    total = N * M
    non_3_count = total - K
    
    # Output: K*N x K*M, filled with 3
    oR, oC = K * N, K * M
    out = [[3] * oC for _ in range(oR)]
    
    # Active tiles: last non_3_count positions in K×K tile grid (row-major)
    active_start = K * K - non_3_count
    for idx in range(active_start, K * K):
        ti, tj = divmod(idx, K)
        # Copy input pattern to this tile position
        for r in range(N):
            for c in range(M):
                out[ti * N + r][tj * M + c] = input_grid[r][c]
    
    return out
