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
    bg_color = color_counts.most_common(1)[0][0]

    # Flood fill to find connected components of background cells.
    # Any component not touching the grid boundary is enclosed → fill with 8.
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] == bg_color:
                component = []
                touches_boundary = False
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    if cr == 0 or cr == rows - 1 or cc == 0 or cc == cols - 1:
                        touches_boundary = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == bg_color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))

                if not touches_boundary:
                    for cr, cc in component:
                        grid[cr][cc] = 8

    return grid
