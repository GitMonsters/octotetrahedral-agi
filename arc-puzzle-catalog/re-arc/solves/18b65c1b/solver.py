from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find complete row (all values non-background)
    complete_row = None
    for r in range(rows):
        if all(grid[r][c] != bg for c in range(cols)):
            complete_row = r
            break

    # Find complete column (all values non-background)
    complete_col = None
    if complete_row is None:
        for c in range(cols):
            if all(grid[r][c] != bg for r in range(rows)):
                complete_col = c
                break

    output = [row[:] for row in grid]

    if complete_row is not None:
        R = complete_row
        for r in range(rows):
            class_values = {}
            for c in range(cols):
                if grid[r][c] != bg:
                    cls = grid[R][c]
                    if cls not in class_values:
                        class_values[cls] = grid[r][c]
            if class_values:
                for c in range(cols):
                    cls = grid[R][c]
                    output[r][c] = class_values.get(cls, bg)

    elif complete_col is not None:
        C = complete_col
        for c in range(cols):
            class_values = {}
            for r in range(rows):
                if grid[r][c] != bg:
                    cls = grid[r][C]
                    if cls not in class_values:
                        class_values[cls] = grid[r][c]
            if class_values:
                for r in range(rows):
                    cls = grid[r][C]
                    output[r][c] = class_values.get(cls, bg)

    return output
