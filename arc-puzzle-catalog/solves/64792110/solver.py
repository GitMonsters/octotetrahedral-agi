def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Two marker pixels (same color, same column, on top & bottom rows) define
    a pair of zigzag / diamond lines that tile across the full grid width.
    
    Each zigzag bounces between top and bottom rows. Together they form an
    X / diamond repeating pattern.
    """
    from collections import Counter

    H = len(grid)
    W = len(grid[0])

    # Detect background (most common color)
    flat = [grid[r][c] for r in range(H) for c in range(W)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find the two marker pixels
    markers = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                markers.append((r, c, grid[r][c]))

    if len(markers) != 2:
        return [row[:] for row in grid]

    (r1, c1, _), (r2, c2, marker_color) = markers
    if r1 > r2:
        r1, c1, r2, c2 = r2, c2, r1, c1

    V = r2 - r1  # vertical span between markers

    # Triangle wave: oscillates 0 → V → 0 with period 2*V
    def triangle(t: int, period: int) -> int:
        t = t % (2 * period)
        return t if t <= period else 2 * period - t

    out = [[bg] * W for _ in range(H)]

    for c in range(W):
        d = abs(c - c1)  # distance from marker column
        t = triangle(d, V)
        row_a = r1 + t   # zigzag from top marker
        row_b = r2 - t   # zigzag from bottom marker
        if 0 <= row_a < H:
            out[row_a][c] = marker_color
        if 0 <= row_b < H:
            out[row_b][c] = marker_color

    return out
