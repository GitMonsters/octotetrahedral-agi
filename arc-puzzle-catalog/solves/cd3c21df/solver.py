from collections import deque


def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)

    shapes = []
    for comp in components:
        min_r = min(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_r = max(r for r, c in comp)
        max_c = max(c for r, c in comp)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        shape = [[0] * w for _ in range(h)]
        for r, c in comp:
            shape[r - min_r][c - min_c] = grid[r][c]
        shapes.append(shape)

    # Convert shapes to tuples for comparison
    shape_keys = [tuple(tuple(row) for row in s) for s in shapes]

    for idx, key in enumerate(shape_keys):
        if shape_keys.count(key) == 1:
            return shapes[idx]

    return shapes[0]


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
