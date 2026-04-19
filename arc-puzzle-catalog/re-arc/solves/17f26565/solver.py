import copy
from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    visited = [[False] * cols for _ in range(rows)]
    objects = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                objects.append(comp)

    output = copy.deepcopy(grid)

    for obj in objects:
        r0 = min(r for r, c in obj)
        c0 = min(c for r, c in obj)
        a = grid[r0][c0]
        b = grid[r0][c0 + 1]
        cv = grid[r0 + 1][c0]
        d = grid[r0 + 1][c0 + 1]

        for br, bc, color in [
            (r0 - 2, c0 - 2, d),
            (r0 - 2, c0 + 2, cv),
            (r0 + 2, c0 - 2, b),
            (r0 + 2, c0 + 2, a),
        ]:
            if color != bg:
                for dr in range(2):
                    for dc in range(2):
                        nr, nc = br + dr, bc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            output[nr][nc] = color

    return output
