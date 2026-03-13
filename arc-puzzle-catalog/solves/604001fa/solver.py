"""
ARC-AGI Task 604001fa Solver

Pattern:
- Input has pairs of (L-tromino made of 7s, shape made of 1s)
- Each L-tromino is 3 cells in a 2x2 bounding box with one corner missing
- The missing corner determines the output color for the paired 1-shape:
    Missing TL → 3 (green)
    Missing TR → 8 (azure)
    Missing BL → 4 (yellow)
    Missing BR → 6 (magenta)
- Output: 7s removed, 1s recolored per their paired L-tromino orientation
"""

import json
from collections import deque
from typing import List, Set, Tuple

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])

    def flood_fill(value: int, connectivity: int) -> List[Set[Tuple[int, int]]]:
        visited: Set[Tuple[int, int]] = set()
        components: List[Set[Tuple[int, int]]] = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == value and (r, c) not in visited:
                    comp: Set[Tuple[int, int]] = set()
                    queue = deque([(r, c)])
                    visited.add((r, c))
                    while queue:
                        cr, cc = queue.popleft()
                        comp.add((cr, cc))
                        if connectivity == 4:
                            nbrs = [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]
                        else:
                            nbrs = [
                                (cr + dr, cc + dc)
                                for dr in (-1, 0, 1)
                                for dc in (-1, 0, 1)
                                if (dr, dc) != (0, 0)
                            ]
                        for nr, nc in nbrs:
                            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == value:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                    components.append(comp)
        return components

    def tromino_color(group: Set[Tuple[int, int]]) -> int:
        """Determine color from which corner of the 2x2 bounding box is missing."""
        min_r = min(r for r, _ in group)
        max_r = max(r for r, _ in group)
        min_c = min(c for _, c in group)
        max_c = max(c for _, c in group)
        if (min_r, min_c) not in group:
            return 3
        if (min_r, max_c) not in group:
            return 8
        if (max_r, min_c) not in group:
            return 4
        if (max_r, max_c) not in group:
            return 6
        return 1  # fallback

    def min_distance(a: Set[Tuple[int, int]], b: Set[Tuple[int, int]]) -> int:
        best = float("inf")
        for r1, c1 in a:
            for r2, c2 in b:
                d = abs(r1 - r2) + abs(c1 - c2)
                if d < best:
                    best = d
        return best

    seven_groups = flood_fill(7, connectivity=4)
    one_groups = flood_fill(1, connectivity=8)

    output = [[0] * cols for _ in range(rows)]

    for sg in seven_groups:
        color = tromino_color(sg)
        best_og = min(one_groups, key=lambda og: min_distance(sg, og))
        for r, c in best_og:
            output[r][c] = color

    return output


if __name__ == "__main__":
    task = json.load(
        open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/604001fa.json")
    )

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
                        print(f"  [{r}][{c}] got {result[r][c]} expected {expected[r][c]}")

    print()
    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            if result == pair["output"]:
                print(f"Test {i}: PASS ✓")
            else:
                print(f"Test {i}: FAIL ✗")
        else:
            print(f"Test {i}: Output generated")
            for row in result:
                print(row)

    print(f"\n{'ALL TRAINING PASSED ✓' if all_pass else 'SOME TRAINING FAILED ✗'}")
