def solve(grid: list[list[int]]) -> list[list[int]]:
    """Draw a rectangle with colored edges and a cross through the center.

    Five colored dots define the structure:
    - Color 2 is the center (cross intersection)
    - The other four become top/bottom/left/right edges of a rectangle
      assigned by extremal row (top/bottom) and column (left/right)
    - Edges are filled with their respective colors
    - A cross of color 5 passes through the center point
    """
    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all non-zero points
    points = {}
    center = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                if grid[r][c] == 2:
                    center = (r, c)
                else:
                    points[grid[r][c]] = (r, c)

    # Assign top/bottom/left/right by extremal position
    by_row = sorted(points.items(), key=lambda x: x[1][0])
    top_color, (top_row, _) = by_row[0]
    bottom_color, (bottom_row, _) = by_row[-1]

    remaining = {k: v for k, v in points.items() if k not in (top_color, bottom_color)}
    by_col = sorted(remaining.items(), key=lambda x: x[1][1])
    left_color, (_, left_col) = by_col[0]
    right_color, (_, right_col) = by_col[-1]

    center_row, center_col = center

    # Top edge
    for c in range(left_col, right_col + 1):
        result[top_row][c] = top_color

    # Bottom edge
    for c in range(left_col, right_col + 1):
        result[bottom_row][c] = bottom_color

    # Left edge
    for r in range(top_row + 1, bottom_row):
        result[r][left_col] = left_color

    # Right edge
    for r in range(top_row + 1, bottom_row):
        result[r][right_col] = right_color

    # Vertical cross line (color 5)
    for r in range(top_row + 1, bottom_row):
        result[r][center_col] = 5

    # Horizontal cross line (color 5)
    for c in range(left_col + 1, right_col):
        result[center_row][c] = 5

    # Center point
    result[center_row][center_col] = 2

    return result
