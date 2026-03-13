"""
Solver for ARC puzzle ac2e8ecf.

Pattern: Each shape is either a "rectangle" (closed border with hole inside,
all bounding-box corners filled) or a "cross" (plus shape, all bounding-box
corners empty). Rectangles are packed to the top of the grid and crosses are
packed to the bottom. Column positions are preserved; only rows shift.
"""

from collections import deque
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    H = len(grid)
    W = len(grid[0])

    # Find connected components via BFS (4-connectivity)
    visited = [[False] * W for _ in range(H)]
    components = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                cells = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                rs = [r for r, c in cells]
                cs = [c for r, c in cells]
                r_min, r_max = min(rs), max(rs)
                c_min, c_max = min(cs), max(cs)
                components.append((color, cells, r_min, r_max, c_min, c_max))

    # Classify: corners filled -> rectangle, else -> cross
    rectangles = []
    crosses = []
    for color, cells, r_min, r_max, c_min, c_max in components:
        cell_set = set(cells)
        corners_filled = all(
            (r, c) in cell_set
            for r, c in [(r_min, c_min), (r_min, c_max), (r_max, c_min), (r_max, c_max)]
        )
        if corners_filled:
            rectangles.append((color, cells, r_min, r_max, c_min, c_max))
        else:
            crosses.append((color, cells, r_min, r_max, c_min, c_max))

    output = [[0] * W for _ in range(H)]

    # Pack rectangles to top, sorted by input top-row then left-col
    rectangles.sort(key=lambda x: (x[2], x[4]))
    for color, cells, r_min, r_max, c_min, c_max in rectangles:
        for new_r_min in range(H):
            dr = new_r_min - r_min
            if all(0 <= r + dr < H and output[r + dr][c] == 0 for r, c in cells):
                for r, c in cells:
                    output[r + dr][c] = color
                break

    # Pack crosses to bottom, sorted by input bottom-row descending
    crosses.sort(key=lambda x: (-x[3], -x[5]))
    for color, cells, r_min, r_max, c_min, c_max in crosses:
        for new_r_max in range(H - 1, -1, -1):
            dr = new_r_max - r_max
            if all(0 <= r + dr < H and output[r + dr][c] == 0 for r, c in cells):
                for r, c in cells:
                    output[r + dr][c] = color
                break

    return output
