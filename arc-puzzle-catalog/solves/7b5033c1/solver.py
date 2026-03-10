from collections import Counter, defaultdict, deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    R, C = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    # Find connected components (4-connectivity, same color)
    visited = [[False] * C for _ in range(R)]
    components: list[tuple[int, set[tuple[int, int]]]] = []

    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                cells: set[tuple[int, int]] = set()
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    cells.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append((color, cells))

    # Map each non-bg cell to its component index
    cell_to_comp: dict[tuple[int, int], int] = {}
    for i, (_color, cells) in enumerate(components):
        for rc in cells:
            cell_to_comp[rc] = i

    # Build adjacency between components (shapes that touch via 4-neighbors)
    adj: defaultdict[int, set[int]] = defaultdict(set)
    for i, (_color, cells) in enumerate(components):
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in cell_to_comp and cell_to_comp[nb] != i:
                    adj[i].add(cell_to_comp[nb])

    # Walk the chain starting from the endpoint with smallest (min_row, min_col)
    endpoints = [i for i in range(len(components)) if len(adj[i]) <= 1]
    start = min(endpoints or range(len(components)),
                key=lambda i: min(components[i][1]))

    order: list[int] = [start]
    seen: set[int] = {start}
    cur = start
    while True:
        nxt = [n for n in adj[cur] if n not in seen]
        if not nxt:
            break
        nxt_id = nxt[0]
        order.append(nxt_id)
        seen.add(nxt_id)
        cur = nxt_id

    output: list[list[int]] = []
    for idx in order:
        color, cells = components[idx]
        output.extend([[color]] * len(cells))
    return output
