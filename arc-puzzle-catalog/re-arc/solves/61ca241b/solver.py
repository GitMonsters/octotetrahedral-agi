def transform(input_grid):
    from collections import Counter

    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most frequent)
    counter = Counter()
    for r in range(rows):
        for c in range(cols):
            counter[grid[r][c]] += 1
    bg = counter.most_common(1)[0][0]

    # Find all 2x2 blocks of non-background color
    blocks = []
    visited = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (r, c) not in visited and grid[r][c] != bg:
                v = grid[r][c]
                if grid[r][c+1] == v and grid[r+1][c] == v and grid[r+1][c+1] == v:
                    blocks.append((r, c))
                    visited.update([(r, c), (r, c+1), (r+1, c), (r+1, c+1)])

    # Place diagonal corner markers for each block
    for r, c in blocks:
        for dr, dc, color in [(-1, -1, 9), (-1, 2, 8), (2, -1, 0), (2, 2, 5)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                grid[nr][nc] = color

    return grid
