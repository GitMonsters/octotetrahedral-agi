def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Each non-background colored region is a bar with one dimension = 3.
    The transformation punches a dotted pattern into the bar's interior:
    - For horizontal bars (height=3): middle row gets alternating bg at odd col offsets.
    - For vertical bars (width=3): middle column gets alternating bg at odd row offsets.
    """
    import numpy as np
    from collections import Counter

    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    bg = Counter(g.flatten().tolist()).most_common(1)[0][0]
    colors = set(g.flatten().tolist()) - {bg}

    for c in colors:
        # Find horizontal runs per row
        runs_by_row: dict[int, list[tuple[int, int]]] = {}
        for r in range(H):
            runs = []
            col = 0
            while col < W:
                if g[r, col] == c:
                    start = col
                    while col < W and g[r, col] == c:
                        col += 1
                    runs.append((start, col - 1))
                else:
                    col += 1
            runs_by_row[r] = runs

        # Group consecutive rows sharing the same run into rectangles
        for r in range(H):
            for run in runs_by_row[r]:
                if r == 0 or run not in runs_by_row.get(r - 1, []):
                    col_start, col_end = run
                    row_start = r
                    row_end = r
                    while row_end + 1 < H and run in runs_by_row.get(row_end + 1, []):
                        row_end += 1

                    h = row_end - row_start + 1
                    w = col_end - col_start + 1

                    if h == 3 and w >= 3:
                        mid_r = row_start + 1
                        for cc in range(col_start, col_end + 1):
                            if (cc - col_start) % 2 == 1:
                                out[mid_r, cc] = bg

                    if w == 3 and h >= 3:
                        mid_c = col_start + 1
                        for rr in range(row_start, row_end + 1):
                            if (rr - row_start) % 2 == 1:
                                out[rr, mid_c] = bg

    return out.tolist()
