"""ARC-AGI solver for task 8dae5dfc: Reverse concentric rectangle layer colors."""

from collections import deque
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                # BFS to find connected component of non-zero cells
                component = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                # Bounding box
                min_r = min(p[0] for p in component)
                max_r = max(p[0] for p in component)
                min_c = min(p[1] for p in component)
                max_c = max(p[1] for p in component)

                # Compute Chebyshev depth from bounding box edge for each cell
                cell_depth = {}
                max_depth = 0
                depth_to_color = {}
                for cr, cc in component:
                    d = min(cr - min_r, max_r - cr, cc - min_c, max_c - cc)
                    cell_depth[(cr, cc)] = d
                    if d > max_depth:
                        max_depth = d
                    if d not in depth_to_color:
                        depth_to_color[d] = grid[cr][cc]

                # Group consecutive depths with same color into logical layers
                layers = []  # list of (color, [depths])
                prev_color = None
                for d in range(max_depth + 1):
                    color = depth_to_color[d]
                    if color != prev_color:
                        layers.append((color, [d]))
                        prev_color = color
                    else:
                        layers[-1][1].append(d)

                # Reverse the layer color ordering
                layer_colors = [l[0] for l in layers]
                reversed_colors = layer_colors[::-1]

                # Build new depth → color mapping
                new_depth_to_color = {}
                for i, (_, depths) in enumerate(layers):
                    for d in depths:
                        new_depth_to_color[d] = reversed_colors[i]

                # Apply reversed colors
                for cr, cc in component:
                    output[cr][cc] = new_depth_to_color[cell_depth[(cr, cc)]]

    return output
