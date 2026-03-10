def solve(grid):
    """Group 8s by shared rows/columns. Fill 0s within each group's bounding box with 2."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Collect all 8-positions
    eights = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    if not eights:
        return out

    # Union-Find to group 8s sharing a row or column
    parent = {}
    for cell in eights:
        parent[cell] = cell

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    # Group by rows: union all 8-cells in the same row
    from collections import defaultdict
    by_row = defaultdict(list)
    by_col = defaultdict(list)
    for r, c in eights:
        by_row[r].append((r, c))
        by_col[c].append((r, c))

    for cells in by_row.values():
        for i in range(1, len(cells)):
            union(cells[0], cells[i])

    for cells in by_col.values():
        for i in range(1, len(cells)):
            union(cells[0], cells[i])

    # Collect components
    components = defaultdict(list)
    for cell in eights:
        components[find(cell)].append(cell)

    # For each component, compute bbox and fill 0s with 2
    for comp in components.values():
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)

        for fr in range(min_r, max_r + 1):
            for fc in range(min_c, max_c + 1):
                if out[fr][fc] == 0:
                    out[fr][fc] = 2

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
