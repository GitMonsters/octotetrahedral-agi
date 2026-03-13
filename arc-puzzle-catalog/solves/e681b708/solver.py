def solve(grid: list[list[int]]) -> list[list[int]]:
    """Replace scattered 1s with the majority color of their region's boundary vertices.

    The grid has structural lines (horizontal/vertical sequences of 1s connecting
    colored vertices). These lines divide the grid into rectangular regions.
    Scattered 1s inside each region are recolored to the majority color of the
    vertices on that region's boundary.
    """
    from collections import Counter, deque

    rows = len(grid)
    cols = len(grid[0])

    # Find vertices (colored markers, value != 0 and != 1)
    vertices = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 1):
                vertices[(r, c)] = grid[r][c]

    # Find structural cells (vertices + 1s on lines between vertices)
    structural = set(vertices.keys())
    vertex_list = list(vertices.keys())
    n = len(vertex_list)
    for i in range(n):
        for j in range(i + 1, n):
            r1, c1 = vertex_list[i]
            r2, c2 = vertex_list[j]
            if r1 == r2:
                lo, hi = min(c1, c2), max(c1, c2)
                if all(grid[r1][c] != 0 for c in range(lo, hi + 1)):
                    for c in range(lo, hi + 1):
                        structural.add((r1, c))
            elif c1 == c2:
                lo, hi = min(r1, r2), max(r1, r2)
                if all(grid[r][c1] != 0 for r in range(lo, hi + 1)):
                    for r in range(lo, hi + 1):
                        structural.add((r, c1))

    # Flood fill non-structural cells to find regions
    visited = [[False] * cols for _ in range(rows)]
    for r, c in structural:
        visited[r][c] = True

    result = [row[:] for row in grid]

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c]:
                region = []
                queue = deque([(r, c)])
                visited[r][c] = True
                boundary_vertices = set()

                while queue:
                    cr, cc = queue.popleft()
                    region.append((cr, cc))

                    # Check 8-neighbors for boundary vertices
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if (nr, nc) in vertices:
                                    boundary_vertices.add((nr, nc))

                    # Expand with 4-connectivity
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not visited[nr][nc] and (nr, nc) not in structural:
                                visited[nr][nc] = True
                                queue.append((nr, nc))

                # Majority color vote
                if boundary_vertices:
                    colors = [vertices[v] for v in boundary_vertices]
                    color_count = Counter(colors)
                    majority_color = color_count.most_common(1)[0][0]
                    for rr, rc in region:
                        if grid[rr][rc] == 1:
                            result[rr][rc] = majority_color

    return result
