"""Solver for ARC-AGI task a04b2602.

Pattern: Green (3) rectangles contain red (2) markers. Each interior 2
gets a 3x3 ring of blue (1) cells. 2s outside rectangles are unchanged.
"""
import json
import copy
from collections import deque
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    output = copy.deepcopy(grid)

    # Find connected components of 3s via BFS
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 3:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(comp)

    # For each component, find bounding box and collect interior 2s
    all_interior_2s = set()
    component_interior = []
    for comp in components:
        min_r = min(r for r, _ in comp)
        max_r = max(r for r, _ in comp)
        min_c = min(c for _, c in comp)
        max_c = max(c for _, c in comp)

        interior_2s = {
            (r, c)
            for r in range(min_r, max_r + 1)
            for c in range(min_c, max_c + 1)
            if grid[r][c] == 2
        }
        all_interior_2s |= interior_2s
        component_interior.append(interior_2s)

    # Place 3x3 ring of 1s around each interior 2 (only protect other interior 2s)
    for interior_2s in component_interior:
        for r2, c2 in interior_2s:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r2 + dr, c2 + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in all_interior_2s:
                        output[nr][nc] = 1

    return output


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/a04b2602.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        match = result == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(result)):
                if result[r] != pair["output"][r]:
                    print(f"  row {r}: got {result[r]}")
                    print(f"       exp {pair['output'][r]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            match = result == pair["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
        else:
            print(f"Test  {i}: produced {len(result)}x{len(result[0])} output")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
