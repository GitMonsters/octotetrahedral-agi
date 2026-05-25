from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Colored markers indicate how to fill regions separated by walls (1s).

    Wall markers (non-0/1 cells with ≥2 adjacent 1-cells) replace wall segments
    and define the wall color for their connected wall component.
    Region markers (non-0/1 cells with <2 adjacent 1-cells) define the fill color
    for their connected 0-region. Every cell gets colored — no 0s or 1s remain.
    """
    rows = len(grid)
    cols = len(grid[0])

    markers: dict[tuple[int, int], int] = {}
    wall_markers: set[tuple[int, int]] = set()
    region_markers: set[tuple[int, int]] = set()

    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v not in (0, 1):
                markers[(r, c)] = v
                adj_ones = 0
                num_neighbors = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        num_neighbors += 1
                        if grid[nr][nc] == 1:
                            adj_ones += 1
                on_boundary = num_neighbors < 4
                if adj_ones >= 2 or (adj_ones >= 1 and on_boundary):
                    wall_markers.add((r, c))
                else:
                    region_markers.add((r, c))

    is_wall = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 or (r, c) in wall_markers:
                is_wall[r][c] = True

    result = [[0] * cols for _ in range(rows)]
    visited = [[False] * cols for _ in range(rows)]

    def bfs(start_r: int, start_c: int, target_is_wall: bool) -> tuple[list, int | None]:
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        cells = [(start_r, start_c)]
        color = None
        while queue:
            r, c = queue.popleft()
            if (r, c) in markers:
                if target_is_wall and (r, c) in wall_markers:
                    color = markers[(r, c)]
                elif not target_is_wall and (r, c) in region_markers:
                    color = markers[(r, c)]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    if is_wall[nr][nc] == target_is_wall:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        cells.append((nr, nc))
        return cells, color

    for r in range(rows):
        for c in range(cols):
            if is_wall[r][c] and not visited[r][c]:
                cells, color = bfs(r, c, True)
                if color is not None:
                    for cr, cc in cells:
                        result[cr][cc] = color

    for r in range(rows):
        for c in range(cols):
            if not is_wall[r][c] and not visited[r][c]:
                cells, color = bfs(r, c, False)
                if color is not None:
                    for cr, cc in cells:
                        result[cr][cc] = color

    return result
