def transform(grid):
    R, C = len(grid), len(grid[0])
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    non_bg = [(r, c, grid[r][c]) for r in range(R) for c in range(C) if grid[r][c] != bg]
    line_color = Counter([v for _, _, v in non_bg]).most_common(1)[0][0]

    v_col = max(range(C), key=lambda c: sum(1 for r in range(R) if grid[r][c] == line_color))
    h_row = max(range(R), key=lambda r: sum(1 for c in range(C) if grid[r][c] == line_color))

    markers = [(r, c) for r, c, v in non_bg if r != h_row and c != v_col]
    N = len(markers)

    mr, mc = markers[0]
    dr = -1 if mr > h_row else 1 if mr < h_row else 0
    dc = -1 if mc > v_col else 1 if mc < v_col else 0

    new_h_row = h_row + dr * N
    new_v_col = v_col + dc * N

    out = [[bg] * C for _ in range(R)]
    for c in range(C):
        out[new_h_row][c] = line_color
    for r in range(R):
        if grid[r][v_col] == line_color or r == h_row:
            out[r][new_v_col] = line_color

    return out
