def solve(grid):
    from collections import deque, Counter

    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]

    def bfs(r, c):
        q = deque([(r, c)])
        visited[r][c] = True
        cells = [(r, c)]
        while q:
            cr, cc = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                        visited[nr][nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
        return cells

    objects = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                cells = bfs(r, c)
                cell_set = set(cells)
                min_r = min(cr for cr, _ in cells)
                max_r = max(cr for cr, _ in cells)
                min_c = min(cc for _, cc in cells)
                max_c = max(cc for _, cc in cells)
                shape = []
                for ri in range(min_r, max_r + 1):
                    row = []
                    for ci in range(min_c, max_c + 1):
                        row.append(grid[ri][ci] if (ri, ci) in cell_set else 0)
                    shape.append(tuple(row))
                objects.append(tuple(shape))

    counts = Counter(objects)
    most_common = counts.most_common(1)[0][0]
    return [list(row) for row in most_common]


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
