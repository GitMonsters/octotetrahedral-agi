import copy

def transform(input_grid):
    grid = copy.deepcopy(input_grid)
    rows = len(grid)
    cols = len(grid[0])

    # Find all 2x2 blocks from the original input
    red_blocks = []
    green_blocks = []
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows - 1):
        for c in range(cols - 1):
            if visited[r][c]:
                continue
            color = input_grid[r][c]
            if color in (2, 3):
                if (input_grid[r][c + 1] == color and
                    input_grid[r + 1][c] == color and
                    input_grid[r + 1][c + 1] == color):
                    visited[r][c] = visited[r][c + 1] = True
                    visited[r + 1][c] = visited[r + 1][c + 1] = True
                    if color == 2:
                        red_blocks.append((r, c))
                    else:
                        green_blocks.append((r, c))

    # Red blocks: diagonal trail up-right from top-right corner
    for r, c in red_blocks:
        tr, tc = r - 1, c + 2
        while 0 <= tr < rows and 0 <= tc < cols:
            grid[tr][tc] = 2
            tr -= 1
            tc += 1

    # Green blocks: diagonal trail down-left from bottom-left corner
    for r, c in green_blocks:
        tr, tc = r + 2, c - 1
        while 0 <= tr < rows and 0 <= tc < cols:
            grid[tr][tc] = 3
            tr += 1
            tc -= 1

    return grid
