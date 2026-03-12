"""ARC-AGI puzzle 00dbd492 solver.

Rule: Each rectangle border of 2s contains an interior with 0s and a single center 2.
Fill the interior 0s with color = 24 // max(interior_width, interior_height).
"""

import copy
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find rectangles of 2s and fill their interiors based on interior size."""
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Find connected components of 2s via BFS
    visited = [[False] * cols for _ in range(rows)]
    components: list[list[tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and not visited[r][c]:
                comp: list[tuple[int, int]] = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 2:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append(comp)

    # Process each rectangle border (large components, skip single-cell interior markers)
    for comp in components:
        if len(comp) <= 1:
            continue

        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)

        interior_h = max_r - min_r - 1
        interior_w = max_c - min_c - 1
        if interior_h <= 0 or interior_w <= 0:
            continue

        fill_color = 24 // max(interior_w, interior_h)

        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if result[r][c] == 0:
                    result[r][c] = fill_color

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
            if status == "FAIL":
                for r, (res_row, exp_row) in enumerate(zip(result, expected)):
                    if res_row != exp_row:
                        print(f"  Row {r}: got {res_row}")
                        print(f"       exp {exp_row}")
        else:
            print(f"Example {i}: no expected output")
