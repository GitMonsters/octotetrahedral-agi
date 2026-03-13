"""
ARC-AGI task 7bb29440 solver.

Pattern: The input contains multiple rectangular regions of non-zero cells
on a zero background. Each region is filled with 1s plus some "special"
values (4s and 6s). The output is the region with the fewest special
(non-0, non-1) cells.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    regions: list[tuple[int, Grid]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                queue = [(r, c)]
                visited[r][c] = True
                min_r = max_r = r
                min_c = max_c = c
                while queue:
                    cr, cc = queue.pop(0)
                    min_r, max_r = min(min_r, cr), max(max_r, cr)
                    min_c, max_c = min(min_c, cc), max(max_c, cc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                subgrid = [row[min_c:max_c + 1] for row in grid[min_r:max_r + 1]]
                specials = sum(1 for row in subgrid for v in row if v not in (0, 1))
                regions.append((specials, subgrid))

    best = min(regions, key=lambda x: x[0])
    return best[1]


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/7bb29440.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        ok = result == pair["output"]
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            print(f"  Expected: {pair['output']}")
            print(f"  Got:      {result}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        print(f"Test {i}: {len(result)}x{len(result[0])}")
        for row in result:
            print(f"  {row}")
        if "output" in pair:
            ok = result == pair["output"]
            print(f"  Match: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
