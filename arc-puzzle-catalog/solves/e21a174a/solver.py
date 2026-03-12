def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    # Find content rows (non-zero)
    content_rows = []
    for r in range(rows):
        if any(grid[r][c] != 0 for c in range(cols)):
            content_rows.append(r)

    if not content_rows:
        return [row[:] for row in grid]

    # Get the dominant non-zero color for each content row
    def row_color(r: int) -> int:
        for c in range(cols):
            if grid[r][c] != 0:
                return grid[r][c]
        return 0

    # Group consecutive content rows by color
    groups: list[list[int]] = []
    for r in content_rows:
        color = row_color(r)
        if groups and row_color(groups[-1][0]) == color:
            groups[-1].append(r)
        else:
            groups.append([r])

    # Reverse the group order, then flatten to get new row mapping
    groups.reverse()
    new_content_rows = [r for group in groups for r in group]

    # Build output
    out = [[0] * cols for _ in range(rows)]
    for dest, src in zip(content_rows, new_content_rows):
        out[dest] = grid[src][:]

    return out
