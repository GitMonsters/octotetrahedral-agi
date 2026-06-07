from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Background = most frequent color
    flat = [cell for row in grid for cell in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Collect all non-background (shape) cells
    shape_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not shape_cells:
        return [row[:] for row in grid]

    shape_color = grid[shape_cells[0][0]][shape_cells[0][1]]

    # The rightmost column across all shape cells is the "reflection axis"
    max_col = max(c for _, c in shape_cells)

    # Start with all-background output
    result = [[bg] * cols for _ in range(rows)]

    # For each row, use the rightmost shape cell to determine the fill
    for r in range(rows):
        row_cols = [c for c in range(cols) if grid[r][c] != bg]
        if not row_cols:
            continue
        c = max(row_cols)           # rightmost shape cell in this row
        if c < max_col:
            fill_start = c + 1
            fill_end = 2 * max_col - c  # inclusive
            for col in range(fill_start, fill_end + 1):
                if 0 <= col < cols:
                    result[r][col] = shape_color

    return result
