import copy

def transform(input_grid):
    grid = copy.deepcopy(input_grid)
    rows, cols = len(grid), len(grid[0])
    to_change = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid[r][c] == 4 and grid[r][c+1] == 4 and
                grid[r+1][c] == 4 and grid[r+1][c+1] == 4):
                to_change.update([(r, c), (r, c+1), (r+1, c), (r+1, c+1)])
    for r, c in to_change:
        grid[r][c] = 1
    return grid
