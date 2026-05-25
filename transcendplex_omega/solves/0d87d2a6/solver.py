import copy
from collections import defaultdict


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Solve ARC task 0d87d2a6.

    Rule: Pairs of 1-markers on grid edges define lines (horizontal if same row,
    vertical if same column). Draw those lines with 1s, and convert any rectangle
    of 2s that the line passes through entirely to 1s. Unaffected rectangles stay as 2s.
    """
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Find all 1-valued cells (markers on grid edges)
    ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]

    # Group by row and column to find line pairs
    row_ones: dict[int, list[int]] = defaultdict(list)
    col_ones: dict[int, list[int]] = defaultdict(list)
    for r, c in ones:
        row_ones[r].append(c)
        col_ones[c].append(r)

    # Build set of all cells on lines (between paired markers)
    line_cells: set[tuple[int, int]] = set()

    for r, cs in row_ones.items():
        if len(cs) == 2:
            c_min, c_max = min(cs), max(cs)
            for c in range(c_min, c_max + 1):
                line_cells.add((r, c))

    for c, rs in col_ones.items():
        if len(rs) == 2:
            r_min, r_max = min(rs), max(rs)
            for r in range(r_min, r_max + 1):
                line_cells.add((r, c))

    # Find connected components of 2s
    visited = [[False] * cols for _ in range(rows)]
    rectangles: list[list[tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and not visited[r][c]:
                component: list[tuple[int, int]] = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 2:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                rectangles.append(component)

    # Convert rectangles intersected by any line to 1s
    for rect in rectangles:
        if any((r, c) in line_cells for r, c in rect):
            for r, c in rect:
                result[r][c] = 1

    # Draw the lines
    for r, c in line_cells:
        result[r][c] = 1

    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/0d87d2a6.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"Test {i}: produced {len(result)}x{len(result[0])}")
