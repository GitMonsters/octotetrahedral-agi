"""
Solver for ARC-AGI task aa4ec2a5.

Pattern: Each input has shapes (connected components of 1s on a background of 4s).
For each shape:
  1. Compute "row-col fill": the intersection of row-wise span fill and column-wise
     span fill. Cells in the original shape become 8; newly filled cells become 6.
  2. If filling adds new cells (holes/concavities), apply 8/6 coloring.
     Otherwise, keep the shape as 1.
  3. Add a 1-cell border of 2 using 8-connectivity (including diagonals).
"""

import json
from collections import deque
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    BG = 4

    output = [row[:] for row in grid]

    # Find connected components of 1s via BFS (4-connectivity)
    visited = [[False] * cols for _ in range(rows)]
    components: list[set[tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                comp: set[tuple[int, int]] = set()
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 1:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append(comp)

    for comp in components:
        # Row span: for each row, the min and max column in the component
        row_span: dict[int, tuple[int, int]] = {}
        # Col span: for each column, the min and max row in the component
        col_span: dict[int, tuple[int, int]] = {}
        for r, c in comp:
            if r in row_span:
                row_span[r] = (min(row_span[r][0], c), max(row_span[r][1], c))
            else:
                row_span[r] = (c, c)
            if c in col_span:
                col_span[c] = (min(col_span[c][0], r), max(col_span[c][1], r))
            else:
                col_span[c] = (r, r)

        # Row-col fill intersection: a cell is filled if it falls within both
        # the row's horizontal span and the column's vertical span.
        filled: set[tuple[int, int]] = set()
        holes: set[tuple[int, int]] = set()
        for r, (cmin, cmax) in row_span.items():
            for c in range(cmin, cmax + 1):
                if c in col_span:
                    rmin, rmax = col_span[c]
                    if rmin <= r <= rmax:
                        filled.add((r, c))
                        if (r, c) not in comp:
                            holes.add((r, c))

        # Apply coloring based on whether holes exist
        if holes:
            for r, c in filled:
                output[r][c] = 8 if (r, c) in comp else 6
            shape_cells = filled
        else:
            shape_cells = comp

        # 8-connected border of 2 around the shape
        for r, c in shape_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in shape_cells and output[nr][nc] == BG:
                        output[nr][nc] = 2

    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/aa4ec2a5.json") as f:
        task = json.load(f)

    pairs = [(i, p, "train") for i, p in enumerate(task.get("train", []))]
    pairs += [(i + len(task.get("train", [])), p, "test") for i, p in enumerate(task.get("test", []))]

    all_pass = True
    for idx, pair, split in pairs:
        result = solve(pair["input"])
        expected = pair["output"]
        if result == expected:
            print(f"Pair {idx} ({split}): PASS")
        else:
            print(f"Pair {idx} ({split}): FAIL")
            all_pass = False
            diffs = 0
            for r in range(max(len(expected), len(result))):
                for c in range(max(len(expected[0]) if expected else 0, len(result[0]) if result else 0)):
                    got = result[r][c] if r < len(result) and c < len(result[0]) else None
                    exp = expected[r][c] if r < len(expected) and c < len(expected[0]) else None
                    if got != exp and diffs < 20:
                        print(f"  ({r},{c}): got {got}, expected {exp}")
                        diffs += 1

    print("ALL PASS" if all_pass else "SOME FAILED")
