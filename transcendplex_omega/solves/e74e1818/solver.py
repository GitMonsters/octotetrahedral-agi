def solve(grid: list[list[int]]) -> list[list[int]]:
    """Reverse the row order within each contiguous single-color row group."""
    import copy
    result = copy.deepcopy(grid)
    rows = len(grid)

    def row_color(row):
        for val in row:
            if val != 0:
                return val
        return 0

    i = 0
    while i < rows:
        color = row_color(grid[i])
        if color == 0:
            i += 1
            continue
        j = i
        while j < rows and row_color(grid[j]) == color:
            j += 1
        # Reverse this group's rows
        group = [grid[r][:] for r in range(i, j)]
        group.reverse()
        for k, r in enumerate(range(i, j)):
            result[r] = group[k]
        i = j

    return result
