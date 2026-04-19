def transform(input_grid):
    import copy
    from collections import Counter, deque

    grid = copy.deepcopy(input_grid)
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1
    bg_color = color_counts.most_common(1)[0][0]

    # Find connected components of non-background cells using BFS
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg_color:
                color = grid[r][c]
                cells = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append((color, cells))

    # For each component, find bounding box top-left and apply checkerboard
    for color, cells in components:
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        for r, c in cells:
            if (r - min_r) % 2 == 1 and (c - min_c) % 2 == 1:
                grid[r][c] = bg_color

    return grid
