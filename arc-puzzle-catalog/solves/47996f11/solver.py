def solve(grid: list[list[int]]) -> list[list[int]]:
    """Restore a transpose-symmetric grid where cells marked 6 are corrupted.

    The grid has three symmetries:
    1. Transpose: out[r][c] == out[c][r]
    2. Vertical flip for inner rows: out[r][c] == out[H+1-r][c] for r in [2, H-1]
    3. Horizontal flip for inner cols: out[r][c] == out[r][W+1-c] for c in [2, W-1]

    Each 6-valued cell is replaced by looking up an equivalent non-6 cell
    under these symmetry transformations.
    """
    H = len(grid)
    W = len(grid[0])
    result = [row[:] for row in grid]

    def get_equivalents(r: int, c: int) -> set[tuple[int, int]]:
        positions = {(r, c), (c, r)}

        if 2 <= r <= H - 1:
            r2 = H + 1 - r
            if 0 <= r2 < H:
                positions.add((r2, c))
                positions.add((c, r2))

        if 2 <= c <= W - 1:
            c2 = W + 1 - c
            if 0 <= c2 < W:
                positions.add((r, c2))
                positions.add((c2, r))

        if 2 <= r <= H - 1 and 2 <= c <= W - 1:
            r2 = H + 1 - r
            c2 = W + 1 - c
            if 0 <= r2 < H and 0 <= c2 < W:
                positions.add((r2, c2))
                positions.add((c2, r2))

        return positions

    for r in range(H):
        for c in range(W):
            if result[r][c] == 6:
                for er, ec in get_equivalents(r, c):
                    if grid[er][ec] != 6:
                        result[r][c] = grid[er][ec]
                        break

    return result
