def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find non-background markers on each edge
    top = {c: input_grid[0][c] for c in range(cols) if input_grid[0][c] != bg}
    bottom = {c: input_grid[rows-1][c] for c in range(cols) if input_grid[rows-1][c] != bg}
    left = {r: input_grid[r][0] for r in range(rows) if input_grid[r][0] != bg}
    right = {r: input_grid[r][cols-1] for r in range(rows) if input_grid[r][cols-1] != bg}

    edges = {}
    if top: edges['top'] = top
    if bottom: edges['bottom'] = bottom
    if left: edges['left'] = left
    if right: edges['right'] = right

    e1, e2 = list(edges.keys())
    c1, c2 = set(edges[e1].values()), set(edges[e2].values())

    # Maroon(9) is always the shift-control edge (Edge B) when colors differ
    if 9 in c2 and 9 not in c1:
        edge_a, edge_b = e1, e2
    elif 9 in c1 and 9 not in c2:
        edge_a, edge_b = e2, e1
    else:
        # Same color: edge with more markers is the source
        edge_a, edge_b = (e1, e2) if len(edges[e1]) >= len(edges[e2]) else (e2, e1)

    source = edges[edge_a]
    shift_set = set(edges[edge_b].keys())
    fill = next(iter(source.values()))
    output = [row[:] for row in input_grid]

    if edge_a in ('top', 'bottom'):
        # Source markers are columns; rays travel along rows
        ray = range(rows) if edge_a == 'top' else range(rows - 1, -1, -1)
        # Shift direction: away from Edge B
        sd = -1 if edge_b == 'right' else 1
        for sc in source:
            cs = 0
            for r in ray:
                if r in shift_set:
                    cs += 1
                c = sc + cs * sd
                if 0 <= c < cols:
                    output[r][c] = fill
    else:
        # Source markers are rows; rays travel along columns
        ray = range(cols) if edge_a == 'left' else range(cols - 1, -1, -1)
        sd = -1 if edge_b == 'bottom' else 1
        for sr in source:
            cs = 0
            for c in ray:
                if c in shift_set:
                    cs += 1
                r = sr + cs * sd
                if 0 <= r < rows:
                    output[r][c] = fill

    return output
