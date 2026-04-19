from collections import Counter


def _find_block_2color(grid, H, W, noise_c):
    """Find the 6x6 block region in 2-color grids using multi-strategy detection."""
    nm = [[grid[r][c] == noise_c for c in range(W)] for r in range(H)]
    cum = [[0] * (W + 1) for _ in range(H + 1)]
    for r in range(H):
        for c in range(W):
            cum[r+1][c+1] = cum[r][c+1] + cum[r+1][c] - cum[r][c] + (1 if nm[r][c] else 0)

    def nin(r1, c1, r2, c2):
        r2, c2 = min(r2, H-1), min(c2, W-1)
        if r1 > r2 or c1 > c2:
            return 0
        return cum[r2+1][c2+1] - cum[r1][c2+1] - cum[r2+1][c1] + cum[r1][c1]

    empty_rows = [r for r in range(H) if nin(r, 0, r, W-1) == 0]
    empty_cols = [c for c in range(W) if nin(0, c, H-1, c) == 0]

    sq6 = [(r1, c1, r1+5, c1+5)
           for r1 in range(H - 5)
           for c1 in range(W - 5)
           if nin(r1, c1, r1+5, c1+5) == 0]

    def score(r1, c1, r2, c2):
        return nin(r1, 0, r2, W-1) + nin(0, c1, H-1, c2) - 2 * nin(r1, c1, r2, c2)

    def touches_edge(r1, c1, r2, c2):
        return r1 == 0 or r2 == H-1 or c1 == 0 or c2 == W-1

    def contains_empty(r1, c1, r2, c2):
        return (any(r1 <= er <= r2 for er in empty_rows) or
                any(c1 <= ec <= c2 for ec in empty_cols))

    # Strategy 1: empty row and empty col at equal offset within block, touching edge
    eq_edge = []
    for er in empty_rows:
        for ec in empty_cols:
            for offset in range(6):
                r1, c1 = er - offset, ec - offset
                if r1 < 0 or c1 < 0:
                    continue
                r2, c2 = r1 + 5, c1 + 5
                if r2 >= H or c2 >= W:
                    continue
                if nin(r1, c1, r2, c2) == 0 and touches_edge(r1, c1, r2, c2):
                    eq_edge.append((r1, c1, r2, c2))
    if eq_edge:
        return max(eq_edge, key=lambda b: score(*b))

    # Strategy 2: highest-score edge-touching block containing an empty row or col
    edge_empty = [b for b in sq6 if touches_edge(*b) and contains_empty(*b)]
    if edge_empty:
        return max(edge_empty, key=lambda b: score(*b))

    # Strategy 3: highest-score edge-touching block
    edge = [b for b in sq6 if touches_edge(*b)]
    if edge:
        return max(edge, key=lambda b: score(*b))

    # Fallback: global highest-score block
    return max(sq6, key=lambda b: score(*b))


def transform(grid):
    """
    Task 373a626b: Noise projection toward a rectangular block.

    Rule: a rectangular block region exists (3rd color filled rectangle in
    3-color grids, or noise-free 6x6 region in 2-color grids). For each row
    in the block's row range, extend noise horizontally toward the block edges.
    For each column in the block's col range, extend noise vertically toward
    the block edges.
    """
    H = len(grid)
    W = len(grid[0])
    cnt = Counter(v for row in grid for v in row)
    bg = cnt.most_common(1)[0][0]
    non_bg = [c for c in cnt if c != bg]

    block_r1 = block_c1 = block_r2 = block_c2 = None
    noise_c = None

    if len(non_bg) == 2:
        # 3-color: find the solid filled rectangle of the non-noise color
        for c in non_bg:
            rows_w = [r for r in range(H) if any(grid[r][col] == c for col in range(W))]
            cols_w = [col for col in range(W) if any(grid[r][col] == c for r in range(H))]
            r1, r2 = min(rows_w), max(rows_w)
            c1, c2 = min(cols_w), max(cols_w)
            solid = all(grid[r][col] == c
                        for r in range(r1, r2+1)
                        for col in range(c1, c2+1))
            if solid:
                block_r1, block_c1, block_r2, block_c2 = r1, c1, r2, c2
                noise_c = [x for x in non_bg if x != c][0]
                break
    else:
        noise_c = non_bg[0]
        block_r1, block_c1, block_r2, block_c2 = _find_block_2color(grid, H, W, noise_c)

    out = [row[:] for row in grid]

    # For each row in block range, fill noise toward block edges horizontally
    for r in range(block_r1, block_r2 + 1):
        left_ns = [c for c in range(block_c1) if grid[r][c] == noise_c]
        if left_ns:
            for c in range(min(left_ns), block_c1):
                out[r][c] = noise_c
        right_ns = [c for c in range(block_c2 + 1, W) if grid[r][c] == noise_c]
        if right_ns:
            for c in range(block_c2 + 1, max(right_ns) + 1):
                out[r][c] = noise_c

    # For each col in block range, fill noise toward block edges vertically
    for col in range(block_c1, block_c2 + 1):
        above_ns = [r for r in range(block_r1) if grid[r][col] == noise_c]
        if above_ns:
            for r in range(min(above_ns), block_r1):
                out[r][col] = noise_c
        below_ns = [r for r in range(block_r2 + 1, H) if grid[r][col] == noise_c]
        if below_ns:
            for r in range(block_r2 + 1, max(below_ns) + 1):
                out[r][col] = noise_c

    return out
