def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find the largest all-zero rectangle and fill its interior with 8."""
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find the largest axis-aligned rectangle of all 0s
    best_area, best_r1, best_c1, best_r2, best_c2 = 0, 0, 0, 0, 0
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] != 0:
                continue
            for r2 in range(r1 + 1, rows):
                if grid[r2][c1] != 0:
                    break
                min_c2 = cols
                for r in range(r1, r2 + 1):
                    # Find max c2 for this row starting from c1
                    for c in range(c1, cols):
                        if grid[r][c] != 0:
                            if c < min_c2:
                                min_c2 = c
                            break
                    else:
                        pass  # entire row to end is 0
                for c2 in range(c1 + 1, min_c2):
                    # Verify all zeros in rectangle [r1..r2, c1..c2]
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    if area > best_area:
                        all_zero = all(
                            grid[r][c] == 0
                            for r in range(r1, r2 + 1)
                            for c in range(c1, c2 + 1)
                        )
                        if all_zero:
                            best_area = area
                            best_r1, best_c1 = r1, c1
                            best_r2, best_c2 = r2, c2

    # Fill interior (inset by 1 on all sides) with 8
    for r in range(best_r1 + 1, best_r2):
        for c in range(best_c1 + 1, best_c2):
            result[r][c] = 8

    return result
