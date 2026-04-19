def transform(grid):
    rows, cols = len(grid), len(grid[0])
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color_cells.setdefault(grid[r][c], []).append((r, c))

    frames = []
    for color, cells in color_cells.items():
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        S = max(h, w)
        rel_cells = set((r - min_r, c - min_c) for r, c in cells)
        frames.append({'color': color, 'size': S, 'h': h, 'w': w, 'rel_cells': rel_cells})

    frames.sort(key=lambda f: f['size'])
    N = frames[-1]['size']

    def find_gap_shift(edge_positions, target_size):
        if not edge_positions:
            return 0
        sorted_pos = sorted(edge_positions)
        gaps = [p for p in range(sorted_pos[0], sorted_pos[-1] + 1) if p not in edge_positions]
        if not gaps:
            return 0
        gap_center = (min(gaps) + max(gaps)) // 2
        return target_size // 2 - gap_center

    out = [[bg] * N for _ in range(N)]

    for frame in reversed(frames):
        S = frame['size']
        h, w = frame['h'], frame['w']
        color = frame['color']
        offset = (N - S) // 2

        top_row_cols = sorted(c for r, c in frame['rel_cells'] if r == 0)
        col_shift = find_gap_shift(top_row_cols, S)

        left_col_rows = sorted(r for r, c in frame['rel_cells'] if c == 0)
        right_col_rows = sorted(r for r, c in frame['rel_cells'] if c == w - 1)
        edge_rows = right_col_rows if len(right_col_rows) >= len(left_col_rows) else left_col_rows
        row_shift = find_gap_shift(edge_rows, S)

        for (r, c) in frame['rel_cells']:
            sr = r + row_shift
            sc = c + col_shift
            for rr, cc in [(sr, sc), (sr, S-1-sc), (S-1-sr, sc), (S-1-sr, S-1-sc)]:
                if 0 <= rr < S and 0 <= cc < S:
                    out[offset + rr][offset + cc] = color

    return out
