"""ARC-AGI task 18419cfa solver.

Pattern: 8-bordered frames (cross-shaped with bumps) enclose a 2-pattern.
The 2-pattern is reflected across the interior center — horizontally if
bumps extend left/right, vertically if bumps extend top/bottom.
"""
import json
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find exterior cells (non-8 reachable from border via 4-connectivity)
    exterior: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid[r][c] != 8:
                exterior.add((r, c))
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in exterior and grid[nr][nc] != 8:
                exterior.add((nr, nc))
                queue.append((nr, nc))

    # Interior = non-8, non-exterior cells
    interior_all: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8 and (r, c) not in exterior:
                interior_all.add((r, c))

    # Find connected components of interior cells
    visited: set[tuple[int, int]] = set()
    regions: list[set[tuple[int, int]]] = []
    for cell in interior_all:
        if cell not in visited:
            region: set[tuple[int, int]] = set()
            q: deque[tuple[int, int]] = deque([cell])
            visited.add(cell)
            while q:
                cr, cc = q.popleft()
                region.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in interior_all and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
            regions.append(region)

    for region in regions:
        twos = [(r, c) for r, c in region if grid[r][c] == 2]
        if not twos:
            continue

        # Interior bounding box
        min_r = min(r for r, c in region)
        max_r = max(r for r, c in region)
        min_c = min(c for r, c in region)
        max_c = max(c for r, c in region)

        center_r = (min_r + max_r) / 2.0
        center_c = (min_c + max_c) / 2.0

        # Center of mass of 2s vs center of region determines reflection axis
        avg_r = sum(r for r, c in twos) / len(twos)
        avg_c = sum(c for r, c in twos) / len(twos)

        if abs(avg_c - center_c) > abs(avg_r - center_r):
            # 2s offset horizontally -> mirror left-right
            for r, c in twos:
                nc = round(2 * center_c - c)
                if (r, nc) in region:
                    result[r][nc] = 2
        else:
            # 2s offset vertically -> mirror top-bottom
            for r, c in twos:
                nr = round(2 * center_r - r)
                if (nr, c) in region:
                    result[nr][c] = 2

    return result


if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/18419cfa.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            status = "PASS" if result == ex["output"] else "FAIL"
            print(f"Test {i}: {status}")
        else:
            print(f"Test {i}: computed")
