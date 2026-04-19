from collections import Counter


def transform(grid):
    grid = [row[:] for row in grid]
    H = len(grid)
    W = len(grid[0])
    N = H - 2  # inner square side length

    # Find mask color (most common value in inner NxN grid)
    inner_vals = []
    for r in range(N):
        for c in range(N):
            inner_vals.append(grid[r][c + 2])
    mask = Counter(inner_vals).most_common(1)[0][0]

    # Fix inner grid using D4 symmetry (8-fold symmetry of a square)
    inner = [[grid[r][c + 2] for c in range(N)] for r in range(N)]
    visited = [[False] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            if visited[r][c]:
                continue
            orbit = list(set([
                (r, c), (c, N - 1 - r), (N - 1 - r, N - 1 - c), (N - 1 - c, r),
                (N - 1 - r, c), (r, N - 1 - c), (c, r), (N - 1 - c, N - 1 - r),
            ]))
            non_mask = set(inner[or_][oc_] for or_, oc_ in orbit if inner[or_][oc_] != mask)
            if len(non_mask) == 1:
                val = non_mask.pop()
                for or_, oc_ in orbit:
                    inner[or_][oc_] = val
            for or_, oc_ in orbit:
                visited[or_][oc_] = True

    for r in range(N):
        for c in range(N):
            grid[r][c + 2] = inner[r][c]

    # Fix border rows (last 2 rows) using V-symmetry on cols 2..W-1
    for br in [H - 2, H - 1]:
        for c in range(N):
            mc = N - 1 - c
            gc, gmc = c + 2, mc + 2
            v1, v2 = grid[br][gc], grid[br][gmc]
            if v1 == mask and v2 != mask:
                grid[br][gc] = v2
            elif v2 == mask and v1 != mask:
                grid[br][gmc] = v1

    # Derive border cols (first 2 cols) from border rows via transpose
    for r in range(N):
        grid[r][0] = grid[H - 1][r + 2]
        grid[r][1] = grid[H - 2][r + 2]

    return grid
