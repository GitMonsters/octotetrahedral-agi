def solve(grid: list[list[int]]) -> list[list[int]]:
    """Fill zero-valued holes using transpose and row/column mirror symmetries.

    The grid is a symmetric matrix (out[r][c] == out[c][r]) that also has
    mirror symmetry: row r == row (n+1-r) for r in [2, n-1].  Zeros mark
    missing cells; each can be recovered from an equivalent position.
    """
    n = len(grid)
    out = [row[:] for row in grid]
    M = n + 1  # mirror axis

    for r in range(n):
        for c in range(n):
            if out[r][c] != 0:
                continue

            mr, mc = M - r, M - c
            # All symmetry-equivalent positions
            positions = [(c, r)]
            if 0 <= mr < n:
                positions += [(mr, c), (c, mr)]
            if 0 <= mc < n:
                positions += [(mc, r), (r, mc)]
            if 0 <= mr < n and 0 <= mc < n:
                positions += [(mr, mc), (mc, mr)]

            for pr, pc in positions:
                if grid[pr][pc] != 0:
                    out[r][c] = grid[pr][pc]
                    break

    return out
