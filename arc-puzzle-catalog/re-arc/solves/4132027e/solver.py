from collections import Counter

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = [row[:] for row in input_grid]
    H = len(grid)
    W = len(grid[0])

    def find_period_1d(seq: list[int]) -> int:
        n = len(seq)
        for p in range(1, n):
            if all(seq[i] == seq[i + p] for i in range(n - p)):
                return p
        return n

    def most_common_period(periods: list[int], max_val: int) -> int:
        count = Counter(periods)
        # Prefer non-trivial periods (> 1 and < max_val)
        candidates = {k: v for k, v in count.items() if 1 < k < max_val}
        if candidates:
            return min(candidates, key=lambda k: (-candidates[k], k))
        if 1 in count:
            return 1
        return max_val

    # Column period: find period of each row, take most common non-trivial
    row_periods = [find_period_1d(grid[r]) for r in range(H)]
    cp = most_common_period(row_periods, W)

    # Row period: find period of each column, take most common non-trivial
    col_data = [[grid[r][c] for r in range(H)] for c in range(W)]
    col_periods = [find_period_1d(col_data[c]) for c in range(W)]
    rp = most_common_period(col_periods, H)

    # Reconstruct tile via majority voting
    tile = [[0] * cp for _ in range(rp)]
    for tr in range(rp):
        for tc in range(cp):
            counts: dict[int, int] = {}
            for r in range(tr, H, rp):
                for c in range(tc, W, cp):
                    v = grid[r][c]
                    counts[v] = counts.get(v, 0) + 1
            tile[tr][tc] = max(counts, key=counts.get)

    # Find bounding box of cells that differ from the tile pattern
    min_r, max_r = H, -1
    min_c, max_c = W, -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] != tile[r % rp][c % cp]:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    if max_r == -1:
        return grid

    # Output the correct tile values for the anomaly rectangle
    return [
        [tile[r % rp][c % cp] for c in range(min_c, max_c + 1)]
        for r in range(min_r, max_r + 1)
    ]
