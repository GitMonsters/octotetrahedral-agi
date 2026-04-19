"""
ARC-AGI Task 43a756f4 Solver

Rule: The input contains 8 shapes scattered on a background, each drawn
with a specific color on a fill-colored rectangular region. Each shape's
non-fill pixels fully cover exactly 1 or 2 edges of the shape's bounding box,
determining its position in a 3x3 output frame:
  TL TC TR
  ML MC MR  (MC is all fill)
  BL BC BR

The output assembles all 8 shapes around the border of a rectangle,
separated by single rows/columns of fill.
"""

from collections import Counter, deque


def transform(grid):
    R, C = len(grid), len(grid[0])
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = [c for c in flat if c != bg]
    fill = Counter(non_bg).most_common(1)[0][0]

    shapes = _find_colored_shapes(grid, R, C, bg, fill)
    shapes += _find_hole_shapes(grid, R, C, bg, fill)

    positions = {s['pos']: s for s in shapes}

    h_top = max(positions[p]['h'] for p in ('TL', 'TC', 'TR') if p in positions)
    h_mid = max(positions[p]['h'] for p in ('ML', 'MR') if p in positions)
    h_bot = max(positions[p]['h'] for p in ('BL', 'BC', 'BR') if p in positions)
    w_left = max(positions[p]['w'] for p in ('TL', 'ML', 'BL') if p in positions)
    w_center = max(positions[p]['w'] for p in ('TC', 'BC') if p in positions)
    w_right = max(positions[p]['w'] for p in ('TR', 'MR', 'BR') if p in positions)

    total_rows = h_top + 1 + h_mid + 1 + h_bot
    total_cols = w_left + 1 + w_center + 1 + w_right
    result = [[fill] * total_cols for _ in range(total_rows)]

    for pos, shape in positions.items():
        sg, sh, sw = shape['grid'], shape['h'], shape['w']
        row_start = {
            'T': 0, 'M': h_top + 1, 'B': h_top + 1 + h_mid + 1
        }[pos[0]]
        col_start = {
            'L': 0, 'C': w_left + 1, 'R': w_left + 1 + w_center + 1
        }[pos[1]]
        sec_h = {'T': h_top, 'M': h_mid, 'B': h_bot}[pos[0]]
        sec_w = {'L': w_left, 'C': w_center, 'R': w_right}[pos[1]]
        dr = 0 if pos[0] in ('T', 'M') else sec_h - sh
        dc = 0 if pos[1] in ('L', 'C') else sec_w - sw

        for r in range(sh):
            for c in range(sw):
                result[row_start + dr + r][col_start + dc + c] = sg[r][c]

    return result


def _classify_position(subgrid, h, w, color):
    top = all(subgrid[0][c] == color for c in range(w))
    bot = all(subgrid[h - 1][c] == color for c in range(w))
    left = all(subgrid[r][0] == color for r in range(h))
    right = all(subgrid[r][w - 1] == color for r in range(h))

    if top and left: return 'TL'
    if top and right: return 'TR'
    if bot and left: return 'BL'
    if bot and right: return 'BR'
    if top: return 'TC'
    if bot: return 'BC'
    if left: return 'ML'
    if right: return 'MR'
    return None


def _find_colored_shapes(grid, R, C, bg, fill):
    visited = [[False] * C for _ in range(R)]
    shapes = []
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] not in (bg, fill):
                color = grid[r][c]
                q = deque([(r, c)])
                visited[r][c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)
                h, w = max_r - min_r + 1, max_c - min_c + 1
                sg = [[fill] * w for _ in range(h)]
                for cr, cc in cells:
                    sg[cr - min_r][cc - min_c] = color
                pos = _classify_position(sg, h, w, color)
                if pos:
                    shapes.append({'color': color, 'grid': sg, 'h': h, 'w': w, 'pos': pos})
    return shapes


def _find_hole_shapes(grid, R, C, bg, fill):
    visited = [[False] * C for _ in range(R)]
    shapes = []
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and grid[r][c] != bg:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = []
                has_non_fill = False
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    if grid[cr][cc] != fill:
                        has_non_fill = True
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                if not has_non_fill:
                    cell_set = set(cells)
                    min_r = min(r for r, c in cells)
                    max_r = max(r for r, c in cells)
                    min_c = min(c for r, c in cells)
                    max_c = max(c for r, c in cells)
                    holes = [(hr, hc) for hr in range(min_r, max_r + 1)
                             for hc in range(min_c, max_c + 1) if (hr, hc) not in cell_set]
                    if holes:
                        hmin_r = min(r for r, c in holes)
                        hmax_r = max(r for r, c in holes)
                        hmin_c = min(c for r, c in holes)
                        hmax_c = max(c for r, c in holes)
                        hh, hw = hmax_r - hmin_r + 1, hmax_c - hmin_c + 1
                        sg = [[fill] * hw for _ in range(hh)]
                        for hr, hc in holes:
                            sg[hr - hmin_r][hc - hmin_c] = bg
                        pos = _classify_position(sg, hh, hw, bg)
                        if pos:
                            shapes.append({'color': bg, 'grid': sg, 'h': hh, 'w': hw, 'pos': pos})
    return shapes