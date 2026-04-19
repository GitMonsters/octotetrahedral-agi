from collections import Counter

def transform(grid: list[list[int]]) -> list[list[int]]:
    """For each column, replace all cells with the column's most frequent color."""
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    for c in range(cols):
        dominant = Counter(grid[r][c] for r in range(rows)).most_common(1)[0][0]
        for r in range(rows):
            output[r][c] = dominant
    return output
