"""Solver for ARC-AGI task 845d6e51.

Pattern: An L-shaped border of 5s encloses a legend in the top-left corner.
The legend contains colored template shapes. Scattered 3-colored shapes appear
elsewhere in the grid. Each 3-shape matches exactly one legend template under
rotation/reflection. Replace every 3-shape with the color of its matching template.
"""

import json
from collections import deque, defaultdict
from typing import List, Set, Tuple, FrozenSet

Grid = List[List[int]]
CellSet = Set[Tuple[int, int]]


def normalize(cells: CellSet) -> Tuple[Tuple[int, int], ...]:
    """Translate shape so min row and col are both 0, return sorted tuple."""
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def canonical(cells: CellSet) -> Tuple[Tuple[int, int], ...]:
    """Return the canonical form of a shape (smallest among all 8 D4 symmetries)."""
    transforms = [
        lambda r, c: (r, c),
        lambda r, c: (c, -r),
        lambda r, c: (-r, -c),
        lambda r, c: (-c, r),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (c, r),
        lambda r, c: (-c, -r),
    ]
    results = []
    for fn in transforms:
        transformed = {fn(r, c) for r, c in cells}
        results.append(normalize(transformed))
    return min(results)


def find_components(grid: Grid, target: int) -> List[CellSet]:
    """Find all 4-connected components of a given value."""
    rows, cols = len(grid), len(grid[0])
    visited: Set[Tuple[int, int]] = set()
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == target and (r, c) not in visited:
                comp: CellSet = set()
                queue = deque([(r, c)])
                visited.add((r, c))
                while queue:
                    cr, cc = queue.popleft()
                    comp.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == target:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                components.append(comp)

    return components


def solve(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])

    # Find the L-shaped 5-border defining the legend rectangle
    border_row = next(r for r in range(rows) if grid[r][0] == 5)
    border_col = next(c for c in range(cols) if grid[0][c] == 5)

    # Extract legend template shapes (non-0, non-5 colors in the legend rectangle)
    legend_colors: dict[int, CellSet] = defaultdict(set)
    for r in range(border_row):
        for c in range(border_col):
            v = grid[r][c]
            if v not in (0, 5):
                legend_colors[v].add((r, c))

    # Build mapping: canonical shape -> color
    canon_to_color: dict[Tuple, int] = {}
    for color, cells in legend_colors.items():
        canon_to_color[canonical(cells)] = color

    # Find all connected components of 3s and replace with matching legend color
    output = [row[:] for row in grid]
    components = find_components(grid, 3)

    for comp in components:
        canon = canonical(comp)
        if canon in canon_to_color:
            color = canon_to_color[canon]
            for r, c in comp:
                output[r][c] = color

    return output


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/845d6e51.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        expected = pair["output"]
        match = result == expected
        all_pass &= match
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Mismatch at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            expected = pair["output"]
            match = result == expected
            all_pass &= match
            print(f"Test {i}: {'PASS' if match else 'FAIL'}")
        else:
            print(f"Test {i}: produced output {len(result)}x{len(result[0])}")

    print(f"\nAll pass: {all_pass}")
