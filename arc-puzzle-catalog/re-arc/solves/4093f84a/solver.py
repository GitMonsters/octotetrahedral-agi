from typing import List


def transform(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find band rows (entire row is 5) and band cols (entire col is 5)
    band_rows = [r for r in range(rows) if all(grid[r][c] == 5 for c in range(cols))]
    band_cols = [c for c in range(cols) if all(grid[r][c] == 5 for r in range(rows))]

    result = [[0] * cols for _ in range(rows)]

    if band_rows:
        # Horizontal band
        r1, r2 = band_rows[0], band_rows[-1]

        # Copy the band itself
        for r in band_rows:
            for c in range(cols):
                result[r][c] = 5

        # For each column, count markers above and below the band
        for c in range(cols):
            n_above = sum(
                1 for r in range(r1) if grid[r][c] not in (0, 5)
            )
            n_below = sum(
                1 for r in range(r2 + 1, rows) if grid[r][c] not in (0, 5)
            )
            # Extend band upward
            for k in range(1, n_above + 1):
                result[r1 - k][c] = 5
            # Extend band downward
            for k in range(1, n_below + 1):
                result[r2 + k][c] = 5

    else:
        # Vertical band
        c1, c2 = band_cols[0], band_cols[-1]

        # Copy the band itself
        for c in band_cols:
            for r in range(rows):
                result[r][c] = 5

        # For each row, count markers left and right of the band
        for r in range(rows):
            n_left = sum(
                1 for c in range(c1) if grid[r][c] not in (0, 5)
            )
            n_right = sum(
                1 for c in range(c2 + 1, cols) if grid[r][c] not in (0, 5)
            )
            # Extend band leftward
            for k in range(1, n_left + 1):
                result[r][c1 - k] = 5
            # Extend band rightward
            for k in range(1, n_right + 1):
                result[r][c2 + k] = 5

    return result
