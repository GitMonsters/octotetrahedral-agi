def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    center = None
    eights = []
    twos = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                center = (r, c)
            elif grid[r][c] == 8:
                eights.append((r, c))
            elif grid[r][c] == 2:
                twos.append((r, c))

    # Compute 8-cell offsets relative to center
    offsets = [(r - center[0], c - center[1]) for r, c in eights]

    # Stamp the 8-pattern at each 2-marker position
    out = [[0] * cols for _ in range(rows)]
    for tr, tc in twos:
        for dr, dc in offsets:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr][nc] = 8

    return out
