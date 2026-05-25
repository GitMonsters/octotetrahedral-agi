def solve(grid: list[list[int]]) -> list[list[int]]:
    """Color each shape (connected 8s) based on number of enclosed holes. 1→1, 2→3, 3→2, 4→4."""
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find connected components of 8s
    visited = [[False] * cols for _ in range(rows)]
    components: list[set[tuple[int, int]]] = []

    def flood_fill(r: int, c: int, val: int, vis: list[list[bool]]) -> set[tuple[int, int]]:
        stack = [(r, c)]
        cells = set()
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if vis[cr][cc] or grid[cr][cc] != val:
                continue
            vis[cr][cc] = True
            cells.add((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                comp = flood_fill(r, c, 8, visited)
                components.append(comp)

    # For each component, count enclosed 0-holes
    hole_map = {1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 6}

    for comp in components:
        # Get bounding box expanded by 1
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        min_r, max_r = min(rs) - 1, max(rs) + 1
        min_c, max_c = min(cs) - 1, max(cs) + 1
        # Clamp
        min_r, min_c = max(0, min_r), max(0, min_c)
        max_r, max_c = min(rows - 1, max_r), min(cols - 1, max_c)

        # Find 0-cells in bbox that are NOT reachable from outside the component
        # Flood fill 0s from border of bbox; remaining 0s are holes
        local_visited = [[False] * cols for _ in range(rows)]
        # Mark component cells as visited (walls)
        for r, c in comp:
            local_visited[r][c] = True

        # Flood fill 0s from the border of the bounding box
        border_queue: list[tuple[int, int]] = []
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if r == min_r or r == max_r or c == min_c or c == max_c:
                    if not local_visited[r][c]:
                        border_queue.append((r, c))

        exterior = set()
        stack = border_queue[:]
        while stack:
            cr, cc = stack.pop()
            if cr < min_r or cr > max_r or cc < min_c or cc > max_c:
                continue
            if local_visited[cr][cc]:
                continue
            if (cr, cc) in exterior:
                continue
            exterior.add((cr, cc))
            local_visited[cr][cc] = True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))

        # Remaining unvisited 0-cells inside bbox are holes
        hole_cells: set[tuple[int, int]] = set()
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if not local_visited[r][c] and grid[r][c] == 0:
                    hole_cells.add((r, c))

        # Count separate hole regions
        hole_visited: set[tuple[int, int]] = set()
        num_holes = 0
        for r, c in hole_cells:
            if (r, c) not in hole_visited:
                num_holes += 1
                q = [(r, c)]
                while q:
                    cr, cc = q.pop()
                    if (cr, cc) in hole_visited:
                        continue
                    if (cr, cc) not in hole_cells:
                        continue
                    hole_visited.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        q.append((cr+dr, cc+dc))

        color = hole_map.get(num_holes, num_holes)
        for r, c in comp:
            result[r][c] = color

    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        expected = ex.get('output')
        if expected:
            status = "PASS" if result == expected else "FAIL"
            print(f"Example {i}: {status}")
        else:
            print(f"Example {i}: no expected output")
