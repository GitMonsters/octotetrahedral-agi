"""
ARC-AGI Puzzle 4b6b68e5 Solver

Pattern: Each grid contains shapes bordered by a single color forming closed/open boundaries.
- Find enclosed regions (flood fill from outside to detect interior cells)
- Count distinct marker colors inside each enclosed region
- If 2+ distinct marker colors exist: fill interior with the most frequent one
- If 0-1 marker colors: leave interior empty (remove markers)
- All stray markers outside any enclosed region are removed
"""

import json
from collections import deque, Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find connected components of same-colored non-zero cells
    visited = [[False] * cols for _ in range(rows)]
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                color = grid[r][c]
                q = deque([(r, c)])
                visited[r][c] = True
                cells: list[tuple[int, int]] = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append((color, cells))

    # Border components are large connected components (size >= 4)
    border_components = []
    all_border_cells: set[tuple[int, int]] = set()
    for color, cells in components:
        if len(cells) >= 4:
            border_components.append((color, cells))
            all_border_cells.update(cells)

    # Output: keep border cells, clear everything else
    output = [[0] * cols for _ in range(rows)]
    for r, c in all_border_cells:
        output[r][c] = grid[r][c]

    # For each border, find enclosed interior via exterior flood fill
    for color, cells in border_components:
        ext_visited = [[False] * cols for _ in range(rows)]
        for r, c in cells:
            ext_visited[r][c] = True

        q = deque()
        for r in range(rows):
            for c_edge in [0, cols - 1]:
                if not ext_visited[r][c_edge]:
                    ext_visited[r][c_edge] = True
                    q.append((r, c_edge))
        for c in range(cols):
            for r_edge in [0, rows - 1]:
                if not ext_visited[r_edge][c]:
                    ext_visited[r_edge][c] = True
                    q.append((r_edge, c))

        while q:
            cr, cc = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not ext_visited[nr][nc]:
                    ext_visited[nr][nc] = True
                    q.append((nr, nc))

        interior = [
            (r, c) for r in range(rows) for c in range(cols) if not ext_visited[r][c]
        ]

        if not interior:
            continue

        # Count marker colors (non-zero, non-border-color, non-border-cell)
        marker_counts: Counter[int] = Counter()
        for r, c in interior:
            val = grid[r][c]
            if val != 0 and val != color and (r, c) not in all_border_cells:
                marker_counts[val] += 1

        # Fill with majority color only if 2+ distinct marker colors
        if len(marker_counts) >= 2:
            fill_color = marker_counts.most_common(1)[0][0]
            for r, c in interior:
                if (r, c) not in all_border_cells:
                    output[r][c] = fill_color

    return output


if __name__ == "__main__":
    with open("/tmp/arc_task_4b6b68e5.json") as f:
        task = json.load(f)

    # Verify against training examples
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")

    # Solve test inputs
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"\nTest {i} output:")
        for row in result:
            print(row)
