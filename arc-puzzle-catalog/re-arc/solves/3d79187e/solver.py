import copy
from collections import deque, Counter


def _get_components(cells):
    remaining = set(cells)
    components = []
    while remaining:
        start = next(iter(remaining))
        comp = set()
        q = deque([start])
        remaining.remove(start)
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb in remaining:
                    remaining.remove(nb)
                    q.append(nb)
        components.append(comp)
    return components


def _find_extent_fn(grid, rows, cols, bg2):
    bg2_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == bg2]
    bg2_min_r = min(r for r, c in bg2_cells)
    bg2_max_r = max(r for r, c in bg2_cells)
    bg2_min_c = min(c for r, c in bg2_cells)
    bg2_max_c = max(c for r, c in bg2_cells)
    if bg2_min_r == 0 and bg2_max_r < rows // 2:
        return lambda r, c: r <= bg2_max_r
    if bg2_max_r == rows - 1 and bg2_min_r >= rows // 2:
        return lambda r, c: r >= bg2_min_r
    if bg2_min_c == 0 and bg2_max_c < cols // 2:
        return lambda r, c: c <= bg2_max_c
    return lambda r, c: c >= bg2_min_c


def transform(grid):
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])
    counts = Counter(grid[r][c] for r in range(rows) for c in range(cols)
                     if grid[r][c] not in (2, 8))
    bg1, bg2 = [k for k, _ in counts.most_common(2)]
    extent_fn = _find_extent_fn(grid, rows, cols, bg2)
    bump_cells = {(r, c) for r in range(rows) for c in range(cols)
                  if grid[r][c] == bg1 and extent_fn(r, c)}
    bump_clusters = _get_components(bump_cells)
    two_cells = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2}
    shapes = _get_components(two_cells)
    placements = []
    used_bumps = set()
    for shape in shapes:
        for bi, bump in enumerate(bump_clusters):
            if bi in used_bumps:
                continue
            bump_frozen = frozenset(bump)
            found = False
            for (sr, sc) in shape:
                for (br, bc) in bump:
                    dr, dc = br - sr, bc - sc
                    placed_in_extent = frozenset(
                        (r + dr, c + dc) for (r, c) in shape
                        if extent_fn(r + dr, c + dc)
                    )
                    if placed_in_extent == bump_frozen:
                        placements.append((shape, dr, dc))
                        used_bumps.add(bi)
                        found = True
                        break
                if found:
                    break
    result = copy.deepcopy(grid)
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 2:
                result[r][c] = bg1
    for (shape, dr, dc) in placements:
        for (r, c) in shape:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr][nc] = 8
    return result
