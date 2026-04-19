from collections import Counter, deque
from typing import List


def transform(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    bg = Counter(cell for row in grid for cell in row).most_common(1)[0][0]

    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                cells = set()
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    cells.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and not visited[nr][nc]
                                and grid[nr][nc] == color):
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append((color, cells))

    # Classify each L-shape by which bounding-box corner is absent:
    #   missing (max_r, max_c) -> TL  (corner at top-left)
    #   missing (max_r, min_c) -> TR  (corner at top-right)
    #   missing (min_r, max_c) -> BL  (corner at bottom-left)
    #   missing (min_r, min_c) -> BR  (corner at bottom-right)
    corners = {}

    for color, cells in components:
        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)

        horiz_len = max_c - min_c + 1
        vert_len = max_r - min_r + 1

        if (max_r, max_c) not in cells:
            ori = "tl"
        elif (max_r, min_c) not in cells:
            ori = "tr"
        elif (min_r, max_c) not in cells:
            ori = "bl"
        elif (min_r, min_c) not in cells:
            ori = "br"
        else:
            continue

        corners[ori] = (color, horiz_len, vert_len)

    tl_color, tl_h, tl_v = corners["tl"]
    tr_color, tr_h, tr_v = corners["tr"]
    bl_color, bl_h, bl_v = corners["bl"]
    br_color, br_h, br_v = corners["br"]

    W = max(tl_h + tr_h, bl_h + br_h)
    H = max(tl_v + bl_v, tr_v + br_v)

    out = [[bg] * W for _ in range(H)]

    for c in range(tl_h):
        out[0][c] = tl_color
    for r in range(tl_v):
        out[r][0] = tl_color

    for c in range(W - tr_h, W):
        out[0][c] = tr_color
    for r in range(tr_v):
        out[r][W - 1] = tr_color

    for c in range(bl_h):
        out[H - 1][c] = bl_color
    for r in range(H - bl_v, H):
        out[r][0] = bl_color

    for c in range(W - br_h, W):
        out[H - 1][c] = br_color
    for r in range(H - br_v, H):
        out[r][W - 1] = br_color

    return out
