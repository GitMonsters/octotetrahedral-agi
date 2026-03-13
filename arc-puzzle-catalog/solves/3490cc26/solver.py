def solve(grid: list[list[int]]) -> list[list[int]]:
    """Connect aligned 2x2 blocks with 7s via greedy nearest-neighbor path from the 2-block.

    All 2x2 blocks (color 8 or 2) are identified. An adjacency graph is built
    where blocks sharing the same column-pair or row-pair are neighbors if no
    other block lies between them on that axis. Starting from the unique 2-block,
    a greedy walk always moves to the nearest unvisited neighbor (by gap size),
    drawing 7s in the space between consecutive blocks along the path.
    """
    import copy
    from collections import defaultdict

    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Find all 2x2 blocks
    blocks = []  # (row, col, color)
    used = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (r, c) not in used and grid[r][c] != 0:
                val = grid[r][c]
                if grid[r][c+1] == val and grid[r+1][c] == val and grid[r+1][c+1] == val:
                    blocks.append((r, c, val))
                    used.update([(r, c), (r, c+1), (r+1, c), (r+1, c+1)])

    # Group blocks by shared column-pair and row-pair
    col_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    row_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (r, c, _) in enumerate(blocks):
        col_groups[(c, c + 1)].append(i)
        row_groups[(r, r + 1)].append(i)

    # Build adjacency list: only consecutive blocks on the same axis
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for indices in col_groups.values():
        indices.sort(key=lambda i: blocks[i][0])
        for k in range(len(indices) - 1):
            i, j = indices[k], indices[k + 1]
            gap = blocks[j][0] - blocks[i][0] - 2
            adj[i].append((j, gap))
            adj[j].append((i, gap))

    for indices in row_groups.values():
        indices.sort(key=lambda i: blocks[i][1])
        for k in range(len(indices) - 1):
            i, j = indices[k], indices[k + 1]
            gap = blocks[j][1] - blocks[i][1] - 2
            adj[i].append((j, gap))
            adj[j].append((i, gap))

    # Find the 2-block (start of path)
    start = next(i for i, (_, _, v) in enumerate(blocks) if v == 2)

    def draw_connection(i: int, j: int) -> None:
        ri, ci, _ = blocks[i]
        rj, cj, _ = blocks[j]
        if ci == cj:  # vertical
            r_lo, r_hi = (min(ri, rj) + 2, max(ri, rj))
            for r in range(r_lo, r_hi):
                result[r][ci] = 7
                result[r][ci + 1] = 7
        else:  # horizontal
            c_lo, c_hi = (min(ci, cj) + 2, max(ci, cj))
            for c in range(c_lo, c_hi):
                result[ri][c] = 7
                result[ri + 1][c] = 7

    # Greedy walk: always pick nearest unvisited neighbor, no backtracking
    visited = {start}
    current = start
    while True:
        neighbors = [(n, gap) for n, gap in adj[current] if n not in visited]
        if not neighbors:
            break
        neighbors.sort(key=lambda x: x[1])
        nxt, _ = neighbors[0]
        draw_connection(current, nxt)
        visited.add(nxt)
        current = nxt

    return result
