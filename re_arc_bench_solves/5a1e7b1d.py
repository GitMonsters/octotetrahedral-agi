from collections import Counter


def transform(grid):
    """Add diagonal corner markers around each 2x2 block of non-background color.

    For each 2x2 block at (r, c)-(r+1, c+1), place:
      3 (green)  at (r-1, c-1) — top-left
      4 (yellow) at (r-1, c+2) — top-right
      9 (maroon) at (r+2, c-1) — bottom-left
      2 (red)    at (r+2, c+2) — bottom-right
    """
    rows = len(grid)
    cols = len(grid[0])

    # Determine background as most frequent color
    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]

    # Find all 2x2 blocks of uniform non-background color
    blocks = []
    used = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (r, c) not in used and grid[r][c] != bg:
                if grid[r][c] == grid[r][c + 1] == grid[r + 1][c] == grid[r + 1][c + 1]:
                    blocks.append((r, c))
                    used.update([(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)])

    # Copy input grid
    output = [row[:] for row in grid]

    # Place corner markers for each block
    for r, c in blocks:
        if r - 1 >= 0 and c - 1 >= 0:
            output[r - 1][c - 1] = 3
        if r - 1 >= 0 and c + 2 < cols:
            output[r - 1][c + 2] = 4
        if r + 2 < rows and c - 1 >= 0:
            output[r + 2][c - 1] = 9
        if r + 2 < rows and c + 2 < cols:
            output[r + 2][c + 2] = 2

    return output
