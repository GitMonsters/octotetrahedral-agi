import numpy as np
from collections import Counter, deque


def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    H, W = grid.shape
    out = grid.copy()

    bg = Counter(grid.flatten()).most_common(1)[0][0]

    # Flood-fill to find connected non-background regions
    visited = np.zeros_like(grid, dtype=bool)
    regions = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))

                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)
                area = (max_r - min_r + 1) * (max_c - min_c + 1)
                is_rect = len(cells) == area

                colors = Counter(grid[cr][cc] for cr, cc in cells)
                regions.append({
                    'bounds': (min_r, min_c, max_r, max_c),
                    'is_rect': is_rect,
                    'cells': cells,
                    'colors': dict(colors),
                    'fill_color': colors.most_common(1)[0][0],
                })

    # Separate the motif (non-rectangular cross pattern) from target rectangles
    motif = None
    rects = []
    for reg in regions:
        if not reg['is_rect']:
            motif = reg
        else:
            rects.append(reg)

    if motif is None:
        max_colors = 0
        for reg in regions:
            if len(reg['colors']) > max_colors:
                max_colors = len(reg['colors'])
                motif = reg
        rects = [r for r in regions if r is not motif]

    # Find motif center at the intersection of the densest row and column
    motif_cells_set = set(motif['cells'])
    row_counts = Counter(r for r, c in motif_cells_set)
    col_counts = Counter(c for r, c in motif_cells_set)
    best_rows = sorted(r for r, cnt in row_counts.items() if cnt == max(row_counts.values()))
    best_cols = sorted(c for c, cnt in col_counts.items() if cnt == max(col_counts.values()))
    center_r = best_rows[len(best_rows) // 2]
    center_c = best_cols[len(best_cols) // 2]

    seed_color = int(grid[center_r][center_c])

    # Build motif as relative offsets from center
    motif_offsets = {}
    for r, c in motif_cells_set:
        motif_offsets[(r - center_r, c - center_c)] = int(grid[r][c])

    # Arm color: most common in motif, fills entire row/column through each seed
    arm_color = Counter(motif_offsets.values()).most_common(1)[0][0]

    # Erase motif from output
    for r, c in motif_cells_set:
        out[r][c] = bg

    # Stamp pattern onto each rectangle at each seed position
    for rect in rects:
        r1, c1, r2, c2 = rect['bounds']
        fill = rect['fill_color']

        seeds = []
        if seed_color != fill:
            seeds = [(cr, cc) for cr, cc in rect['cells'] if grid[cr][cc] == seed_color]
        else:
            # Seed invisible (same as fill): use rectangle center
            seeds = [(r1 + (r2 - r1 + 1) // 2, c1 + (c2 - c1 + 1) // 2)]

        for sr, sc in seeds:
            # Extend arm color across entire seed row and column within rectangle
            for c in range(c1, c2 + 1):
                out[sr][c] = arm_color
            for r in range(r1, r2 + 1):
                out[r][sc] = arm_color
            # Overlay non-arm motif cells at their offsets, clipped to rectangle
            for (dr, dc), color in motif_offsets.items():
                if color != arm_color:
                    nr, nc = sr + dr, sc + dc
                    if r1 <= nr <= r2 and c1 <= nc <= c2:
                        out[nr][nc] = color

    return out.tolist()
