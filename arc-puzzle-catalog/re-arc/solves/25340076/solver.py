def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    def is_uniform_row(r):
        return len(set(grid[r])) == 1

    def is_uniform_col(c):
        return len(set(grid[r][c] for r in range(rows))) == 1

    def get_col(c):
        return [grid[r][c] for r in range(rows)]

    result = [row[:] for row in grid]

    # Key at bottom
    if not is_uniform_row(rows - 1) and is_uniform_row(rows - 2):
        key = grid[rows - 1]
        sep_row = rows - 2
        n = len(key)
        for r in range(sep_row):
            dist = sep_row - 1 - r
            result[r] = [key[dist % n]] * cols
        return result

    # Key at top
    if not is_uniform_row(0) and is_uniform_row(1):
        key_read = grid[0][::-1]
        sep_row = 1
        n = len(key_read)
        for r in range(sep_row + 1, rows):
            dist = r - sep_row - 1
            result[r] = [key_read[dist % n]] * cols
        return result

    # Key at right
    if not is_uniform_col(cols - 1) and is_uniform_col(cols - 2):
        key_read = get_col(cols - 1)[::-1]
        sep_col = cols - 2
        n = len(key_read)
        for c in range(sep_col):
            dist = sep_col - 1 - c
            color = key_read[dist % n]
            for r in range(rows):
                result[r][c] = color
        return result

    # Key at left
    if not is_uniform_col(0) and is_uniform_col(1):
        key_read = get_col(0)
        sep_col = 1
        n = len(key_read)
        for c in range(sep_col + 1, cols):
            dist = c - sep_col - 1
            color = key_read[dist % n]
            for r in range(rows):
                result[r][c] = color
        return result

    return result
