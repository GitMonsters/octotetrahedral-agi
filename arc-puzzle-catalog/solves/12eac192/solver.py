def solve(grid):
    # Rule: For each non-zero color, find connected components (4-connectivity).
    # Components of size >= 3 keep their color. Components of size < 3 become color 3.
    rows, cols = len(grid), len(grid[0])
    result = [[grid[r][c] for c in range(cols)] for r in range(rows)]

    # Find all non-zero colors
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                colors.add(grid[r][c])

    for color in colors:
        # Find connected components of this color
        visited = [[False] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and not visited[r][c]:
                    # BFS to find component
                    component = []
                    queue = [(r, c)]
                    visited[r][c] = True
                    while queue:
                        cr, cc = queue.pop(0)
                        component.append((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                    if len(component) < 3:
                        for cr, cc in component:
                            result[cr][cc] = 3
    return result

if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
