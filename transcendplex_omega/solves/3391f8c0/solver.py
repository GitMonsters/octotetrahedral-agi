"""
ARC-AGI Puzzle 3391f8c0 Solver

Rule: Two non-background colors A and B each form shapes on the grid.
All instances of color A share the same pattern; same for color B.
The transformation swaps shapes and colors:
  - Each instance of color A's shape is replaced by color B's pattern drawn in color B
  - Each instance of color B's shape is replaced by color A's pattern drawn in color A
Patterns are aligned by bounding-box top-left corner.
"""

import json
from collections import deque
from pathlib import Path


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find the two non-background colors
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                colors.add(grid[r][c])
    colors = sorted(colors)
    assert len(colors) == 2
    color_a, color_b = colors

    def find_components(color: int) -> list[list[tuple[int, int]]]:
        """Find connected components (8-connectivity) for a given color."""
        visited: set[tuple[int, int]] = set()
        components = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r, c) not in visited:
                    comp: list[tuple[int, int]] = []
                    queue = deque([(r, c)])
                    visited.add((r, c))
                    while queue:
                        cr, cc = queue.popleft()
                        comp.append((cr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = cr + dr, cc + dc
                                if (0 <= nr < rows and 0 <= nc < cols
                                        and (nr, nc) not in visited
                                        and grid[nr][nc] == color):
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                    components.append(comp)
        return components

    def get_pattern(comp: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        """Get pattern as offsets from bounding-box top-left."""
        min_r = min(r for r, c in comp)
        min_c = min(c for r, c in comp)
        return frozenset((r - min_r, c - min_c) for r, c in comp)

    def get_top_left(comp: list[tuple[int, int]]) -> tuple[int, int]:
        min_r = min(r for r, c in comp)
        min_c = min(c for r, c in comp)
        return (min_r, min_c)

    comps_a = find_components(color_a)
    comps_b = find_components(color_b)

    # Get canonical pattern from first component of each color
    pattern_a = get_pattern(comps_a[0])
    pattern_b = get_pattern(comps_b[0])

    # Build output
    output = [[0] * cols for _ in range(rows)]

    # At each color_a position, place pattern_b in color_b
    for comp in comps_a:
        tr, tc = get_top_left(comp)
        for dr, dc in pattern_b:
            output[tr + dr][tc + dc] = color_b

    # At each color_b position, place pattern_a in color_a
    for comp in comps_b:
        tr, tc = get_top_left(comp)
        for dr, dc in pattern_a:
            output[tr + dr][tc + dc] = color_a

    return output


if __name__ == "__main__":
    puzzle_path = Path.home() / "ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/3391f8c0.json"
    with open(puzzle_path) as f:
        data = json.load(f)

    all_pass = True
    for i, pair in enumerate(data["train"]):
        inp = pair["input"]
        expected = pair["output"]
        got = solve(inp)
        if got == expected:
            print(f"Training example {i}: PASS")
        else:
            all_pass = False
            print(f"Training example {i}: FAIL")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if got[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): expected {expected[r][c]}, got {got[r][c]}")

    if all_pass:
        print("\nAll training examples passed!")
    else:
        print("\nSome examples FAILED.")
