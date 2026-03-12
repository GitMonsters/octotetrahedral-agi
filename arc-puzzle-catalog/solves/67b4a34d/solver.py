def solve(grid: list[list[int]]) -> list[list[int]]:
    """Recover the 4x4 region masked by 3s using 180° rotational symmetry."""
    n = len(grid)
    # Find the 4x4 block of 3s
    mask_r = mask_c = -1
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 3:
                mask_r, mask_c = r, c
                break
        if mask_r >= 0:
            break

    output = []
    for r in range(mask_r, mask_r + 4):
        row = []
        for c in range(mask_c, mask_c + 4):
            # 180° rotation: (r,c) -> (n-1-r, n-1-c)
            row.append(grid[n - 1 - r][n - 1 - c])
        output.append(row)
    return output
