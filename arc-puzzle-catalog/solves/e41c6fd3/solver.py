def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find all non-zero colors
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                colors.add(grid[r][c])

    # For each color, find top row
    top_row = {}
    for color in colors:
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    top_row[color] = r
                    break
            if color in top_row:
                break

    # Color 8 is the anchor — all shapes align to its top row
    anchor = top_row[8]

    # Build output: shift each color's pixels vertically to align with anchor
    output = [[0] * cols for _ in range(rows)]
    for color in colors:
        dy = anchor - top_row[color]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    nr = r + dy
                    if 0 <= nr < rows:
                        output[nr][c] = color

    return output
