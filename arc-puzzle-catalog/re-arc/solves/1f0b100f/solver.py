from collections import Counter
from math import gcd


def trace_bouncing(sr: int, sc: int, dr: int, dc: int, rows: int, cols: int, steps: int) -> set:
    """Trace a bouncing diagonal ray for a given number of steps."""
    positions = set()
    r, c = sr, sc
    for _ in range(steps + 1):
        positions.add((r, c))
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= rows:
            dr = -dr
            nr = r + dr
        if nc < 0 or nc >= cols:
            dc = -dc
            nc = c + dc
        r, c = nr, nc
    return positions


def transform(grid: list[list[int]]) -> list[list[int]]:
    R = len(grid)
    C = len(grid[0])
    cnt = Counter(v for row in grid for v in row)
    corners = [(0, 0), (0, C - 1), (R - 1, 0), (R - 1, C - 1)]

    if len(cnt) == 1:
        # Uniform grid: signal pattern from grid-dimension modular formula
        color = next(iter(cnt))
        min_d = min(R - 1, C - 1)
        max_d = max(R - 1, C - 1)
        g = gcd(min_d, max_d) if min_d > 0 else 1

        if g == 1:
            mod = min_d
            offset = max_d % min_d
            s_sum = {offset}
            s_diff = {(mod - offset) % mod}
        else:
            mod = 2 * min_d
            s_sum = {k * g for k in range(min_d // g + 1)}
            s_diff = s_sum

        out = [[9] * C for _ in range(R)]
        for r in range(R):
            for c in range(C):
                if (r + c) % mod in s_sum or (r - c) % mod in s_diff:
                    out[r][c] = color
        return out

    # Non-uniform: bounce diagonals from signal-colored corners
    bg = cnt.most_common(1)[0][0]
    sig_color = next(v for v in cnt if v != bg)
    marked = [(r, c) for r, c in corners if grid[r][c] == sig_color]

    max_steps = max(R - 1, C - 1)
    signal_cells = set()
    for mr, mc in marked:
        dr = 1 if mr == 0 else -1
        dc = 1 if mc == 0 else -1
        signal_cells |= trace_bouncing(mr, mc, dr, dc, R, C, max_steps)

    out = [[9] * C for _ in range(R)]
    for r, c in signal_cells:
        out[r][c] = sig_color
    return out
