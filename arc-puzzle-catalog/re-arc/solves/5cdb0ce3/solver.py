from collections import Counter, deque


def _components(grid, background):
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] == background:
                continue
            color = grid[r][c]
            queue = deque([(r, c)])
            seen[r][c] = True
            cells = []
            while queue:
                x, y = queue.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx][ny] and grid[nx][ny] == color:
                        seen[nx][ny] = True
                        queue.append((nx, ny))
            yield cells


def _component_type(cells):
    cell_set = set(cells)
    neighbors = {
        cell: [
            (cell[0] + dx, cell[1] + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if (cell[0] + dx, cell[1] + dy) in cell_set
        ]
        for cell in cell_set
    }
    degrees = {cell: len(adj) for cell, adj in neighbors.items()}
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]

    if any(degree > 2 for degree in degrees.values()) or len(endpoints) != 2:
        return 3

    start = min(endpoints)
    path = [start]
    prev = None
    cur = start
    while True:
        nxt = [cell for cell in neighbors[cur] if cell != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)

    turns = 0
    for i in range(len(path) - 2):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        r3, c3 = path[i + 2]
        if (r2 - r1, c2 - c1) != (r3 - r2, c3 - c2):
            turns += 1

    return 6 if turns <= 1 else 9


def transform(grid):
    background = Counter(value for row in grid for value in row).most_common(1)[0][0]
    result = [row[:] for row in grid]
    for cells in _components(grid, background):
        new_color = _component_type(cells)
        for r, c in cells:
            result[r][c] = new_color
    return result
