"""
ARC-AGI Task 7d1f7ee8 Solver

Pattern: The grid contains nested hollow rectangles. Every non-zero cell
inside a hollow rectangle's interior gets recolored to the color of the
outermost enclosing rectangle.
"""

import json
from collections import deque
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    # Find connected components of non-zero colors via BFS
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                comp = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append((color, comp))

    # Identify which components form hollow rectangles
    rectangles = []
    for color, comp in components:
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)

        # Need at least 3x3 to be hollow
        if max_r - min_r < 2 or max_c - min_c < 2:
            continue

        # Build expected border cell set
        expected = set()
        for cc in range(min_c, max_c + 1):
            expected.add((min_r, cc))
            expected.add((max_r, cc))
        for rr in range(min_r + 1, max_r):
            expected.add((rr, min_c))
            expected.add((rr, max_c))

        if set(comp) == expected:
            rectangles.append((color, min_r, max_r, min_c, max_c))

    # Sort by area descending so outermost rectangle is checked first
    rectangles.sort(key=lambda x: (x[2] - x[1]) * (x[4] - x[3]), reverse=True)

    # Replace each non-zero cell with the color of its outermost enclosing rectangle
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                for color, min_r, max_r, min_c, max_c in rectangles:
                    if min_r < r < max_r and min_c < c < max_c:
                        output[r][c] = color
                        break

    return output


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/7d1f7ee8.json"))

    # Test on training examples
    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        expected = pair["output"]
        if result == expected:
            print(f"Train {i}: PASS ✓")
        else:
            all_pass = False
            print(f"Train {i}: FAIL ✗")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

    # Test on test examples
    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            expected = pair["output"]
            if result == expected:
                print(f"Test  {i}: PASS ✓")
            else:
                all_pass = False
                print(f"Test  {i}: FAIL ✗")
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        if result[r][c] != expected[r][c]:
                            print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
        else:
            print(f"Test  {i}: (no expected output) Result grid {len(result)}x{len(result[0])}")

    if all_pass:
        print("\nALL TESTS PASSED ✓")
