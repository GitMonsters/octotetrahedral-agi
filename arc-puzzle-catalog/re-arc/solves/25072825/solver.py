def transform(input_grid):
    from collections import Counter

    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1
    bg = color_counts.most_common(1)[0][0]

    # Find connected components of non-background cells
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                objects.append((color, component))

    # For each object, punch holes where both row and col offset are odd
    for color, component in objects:
        min_r = min(r for r, c in component)
        min_c = min(c for r, c in component)
        for r, c in component:
            if (r - min_r) % 2 == 1 and (c - min_c) % 2 == 1:
                grid[r][c] = bg

    return grid
