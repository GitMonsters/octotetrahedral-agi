"""ARC puzzle b942fd60 - Line bounce/reflection from red seed.

Rule: The single red (2) cell sends horizontal lines that bounce off colored
obstacles. At each obstacle-stopped endpoint, perpendicular lines extend in
both directions. Boundary endpoints don't bounce. Lines pass through
previously marked cells (checking original grid for obstacles).
"""

def solve(grid):
    from collections import deque
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Find seed (color 2)
    seed = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                seed = (r, c)
                break
        if seed:
            break
    if not seed:
        return out

    sr, sc = seed
    visited = set()
    queue = deque()

    def extend(r, c, dr, dc):
        cells = []
        nr, nc = r + dr, c + dc
        while 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            cells.append((nr, nc))
            out[nr][nc] = 2
            nr += dr
            nc += dc
        obstacle = (0 <= nr < rows and 0 <= nc < cols)
        return cells, obstacle

    # Seed extends horizontally
    cells_r, obs_r = extend(sr, sc, 0, 1)
    if obs_r and cells_r:
        queue.append((cells_r[-1], 'V'))
    cells_l, obs_l = extend(sr, sc, 0, -1)
    if obs_l and cells_l:
        queue.append((cells_l[-1], 'V'))

    while queue:
        (r, c), next_dir = queue.popleft()
        if ((r, c), next_dir) in visited:
            continue
        visited.add(((r, c), next_dir))

        if next_dir == 'V':
            for dr, dc in [(-1, 0), (1, 0)]:
                cells, obs = extend(r, c, dr, dc)
                if obs and cells:
                    queue.append((cells[-1], 'H'))
        else:
            for dr, dc in [(0, -1), (0, 1)]:
                cells, obs = extend(r, c, dr, dc)
                if obs and cells:
                    queue.append((cells[-1], 'V'))

    return out
