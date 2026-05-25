"""Solver for ARC-AGI task 7ed72f31.

Pattern: Each colored shape is 8-connected to a group of 2-cells that acts as
a reflection axis.  A single 2-cell triggers point reflection (180° rotation);
a horizontal line of 2s mirrors across its row; a vertical line mirrors across
its column.  The reflected copy is painted in the same color.
"""

import copy
import json
from collections import Counter, deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    output = copy.deepcopy(grid)

    # Background = most frequent value
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    # Collect non-background cells
    non_bg = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg}

    # 8-connected components of non-background cells
    visited: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []
    for seed in non_bg:
        if seed in visited:
            continue
        comp: set[tuple[int, int]] = set()
        q = deque([seed])
        visited.add(seed)
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in non_bg and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        components.append(comp)

    for comp in components:
        twos = {p for p in comp if grid[p[0]][p[1]] == 2}
        shape = {p for p in comp if grid[p[0]][p[1]] != 2}
        if not twos or not shape:
            continue

        two_rows = {r for r, _ in twos}
        two_cols = {c for _, c in twos}

        if len(twos) == 1:
            # Point reflection
            r0, c0 = next(iter(twos))
            for r, c in shape:
                nr, nc = 2 * r0 - r, 2 * c0 - c
                if 0 <= nr < rows and 0 <= nc < cols:
                    output[nr][nc] = grid[r][c]
        elif len(two_rows) == 1:
            # Horizontal line → mirror across row
            r0 = next(iter(two_rows))
            for r, c in shape:
                nr = 2 * r0 - r
                if 0 <= nr < rows:
                    output[nr][c] = grid[r][c]
        elif len(two_cols) == 1:
            # Vertical line → mirror across column
            c0 = next(iter(two_cols))
            for r, c in shape:
                nc = 2 * c0 - c
                if 0 <= nc < cols:
                    output[r][nc] = grid[r][c]

    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/7ed72f31.json") as f:
        data = json.load(f)

    all_pass = True
    for section in ("train", "test"):
        for i, pair in enumerate(data[section]):
            result = solve(pair["input"])
            expected = pair["output"]
            ok = result == expected
            print(f"{section} {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        if result[r][c] != expected[r][c]:
                            print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

    raise SystemExit(0 if all_pass else 1)
