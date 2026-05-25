"""Solver for ARC puzzle 696d4842.

Each shape is a path-like polyomino with two endpoints. A nearby isolated dot
determines which endpoint's arm extends toward it. The opposite arm gets
recolored with the dot's color (same number of cells as added).
"""

from collections import deque
from typing import List, Tuple, Set, Optional


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Find connected components of non-zero cells
    visited = [[False] * cols for _ in range(rows)]
    components: List[List[Tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                comp = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append(comp)

    # Separate shapes (size > 1) and dots (size == 1)
    shapes = []
    dots = []
    for comp in components:
        if len(comp) == 1:
            r, c = comp[0]
            dots.append((grid[r][c], (r, c)))
        else:
            color = grid[comp[0][0]][comp[0][1]]
            shapes.append((color, comp))

    used_dots: Set[Tuple[int, int]] = set()

    for shape_color, shape in shapes:
        shape_set = set(shape)

        # Find endpoints: cells with exactly 1 neighbor in the shape
        endpoints = []
        for r, c in shape:
            count = sum(
                1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if (r + dr, c + dc) in shape_set
            )
            if count == 1:
                endpoints.append((r, c))

        if len(endpoints) != 2:
            continue

        # For each endpoint, find its extension direction
        ep_data = []
        for ep in endpoints:
            r, c = ep
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in shape_set:
                    arm_dir = (dr, dc)
                    break
            ext_dir = (-arm_dir[0], -arm_dir[1])
            ep_data.append({'pos': ep, 'arm_dir': arm_dir, 'ext_dir': ext_dir})

        # Try to pair an endpoint with a dot along its extension line
        matched = False
        for i, ed in enumerate(ep_data):
            if matched:
                break
            ep = ed['pos']
            ext_r, ext_c = ed['ext_dir']

            for dot_color, dot_pos in dots:
                if dot_pos in used_dots:
                    continue
                dr = dot_pos[0] - ep[0]
                dc = dot_pos[1] - ep[1]

                # Check dot is aligned with extension direction
                if ext_r == 0 and ext_c != 0:
                    if dr != 0 or dc == 0:
                        continue
                    if (dc > 0) != (ext_c > 0):
                        continue
                elif ext_c == 0 and ext_r != 0:
                    if dc != 0 or dr == 0:
                        continue
                    if (dr > 0) != (ext_r > 0):
                        continue
                else:
                    continue

                n_new = abs(dr) + abs(dc) - 1

                # Verify path is clear
                clear = True
                cr, cc = ep
                for _ in range(n_new):
                    cr += ext_r
                    cc += ext_c
                    if grid[cr][cc] != 0:
                        clear = False
                        break
                if not clear:
                    continue

                # Extend from endpoint toward dot
                cr, cc = ep
                for _ in range(n_new):
                    cr += ext_r
                    cc += ext_c
                    out[cr][cc] = shape_color

                # Trace path from opposite endpoint and recolor n_new cells
                other_ep = ep_data[1 - i]['pos']
                path = _trace_path(shape_set, other_ep)
                for j in range(min(n_new, len(path))):
                    pr, pc = path[j]
                    out[pr][pc] = dot_color

                used_dots.add(dot_pos)
                matched = True
                break

    return out


def _trace_path(
    shape_set: Set[Tuple[int, int]], start: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Trace path through shape from start, returning ordered cells."""
    path = [start]
    visited = {start}
    current = start
    while True:
        r, c = current
        next_cell = None
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in shape_set and (nr, nc) not in visited:
                next_cell = (nr, nc)
                break
        if next_cell is None:
            break
        path.append(next_cell)
        visited.add(next_cell)
        current = next_cell
    return path
