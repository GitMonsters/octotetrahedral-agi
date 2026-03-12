def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    # Find gap columns: positions where a 0 has the same non-zero color on both sides
    gap_columns: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(1, cols - 1):
            if grid[r][c] == 0:
                left, right = grid[r][c - 1], grid[r][c + 1]
                if left != 0 and right != 0 and left == right:
                    gap_columns.setdefault(c, []).append((r, left))

    for c, h_lines in gap_columns.items():
        h_lines.sort()

        # Find bottom vertical stub color below last horizontal line
        last_row = h_lines[-1][0]
        bottom_color = 0
        for r in range(last_row + 1, rows):
            if grid[r][c] != 0:
                bottom_color = grid[r][c]
                break

        # Fill zones top-to-bottom
        for i, (h_row, h_color) in enumerate(h_lines):
            start = 0 if i == 0 else h_lines[i - 1][0] + 1
            for r in range(start, h_row + 1):
                output[r][c] = h_color

        # Fill below last horizontal line with bottom stub color
        if bottom_color != 0:
            for r in range(last_row + 1, rows):
                output[r][c] = bottom_color

    return output
