from collections import Counter, deque


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]

    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)

    output = [[bg] * cols for _ in range(rows)]
    for comp in components:
        # U-shapes (2 turns) → green(3), everything else → blue(1)
        out_color = 3 if _count_turns(comp) == 2 else 1
        for r, c in comp:
            output[r][c] = out_color

    return output


def _count_turns(comp):
    """Count direction changes along a non-branching shape's path."""
    comp_set = set(comp)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    endpoints = []
    for r, c in comp:
        if sum(1 for dr, dc in dirs if (r + dr, c + dc) in comp_set) == 1:
            endpoints.append((r, c))

    if len(endpoints) != 2:
        return -1

    path = [endpoints[0]]
    seen = {endpoints[0]}
    while True:
        r, c = path[-1]
        nxt = [(r + dr, c + dc) for dr, dc in dirs
               if (r + dr, c + dc) in comp_set and (r + dr, c + dc) not in seen]
        if not nxt:
            break
        path.append(nxt[0])
        seen.add(nxt[0])

    if len(path) != len(comp):
        return -2

    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2:
            turns += 1
    return turns
