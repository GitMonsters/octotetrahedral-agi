from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    result = [row[:] for row in grid]

    cols_interior_bg = all(grid[r][c] == bg for r in range(rows) for c in range(1, cols - 1))
    rows_interior_bg = all(grid[r][c] == bg for r in range(1, rows - 1) for c in range(cols))

    if cols_interior_bg:
        mid = cols // 2
        for r in range(rows):
            left = grid[r][0]
            right = grid[r][cols - 1]
            if left == bg and right == bg:
                continue
            for c in range(mid):
                result[r][c] = left
            result[r][mid] = 0
            for c in range(mid + 1, cols):
                result[r][c] = right
    elif rows_interior_bg:
        mid = rows // 2
        for c in range(cols):
            top = grid[0][c]
            bottom = grid[rows - 1][c]
            if top == bg and bottom == bg:
                continue
            for r in range(mid):
                result[r][c] = top
            result[mid][c] = 0
            for r in range(mid + 1, rows):
                result[r][c] = bottom

    return result
