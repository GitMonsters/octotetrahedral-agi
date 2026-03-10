"""
ARC-AGI solver for task 6ad5bdfd

Transformation: A wall of 2s spans one full row or column at the grid edge.
All colored objects (connected components of same non-0, non-2 color) slide
toward the wall like gravity, preserving shape and stacking against each other.
"""
import json
from collections import deque
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find the wall of 2s (a full row or column)
    wall_type = None
    wall_idx = None

    for r in range(rows):
        if all(grid[r][c] == 2 for c in range(cols)):
            wall_idx = r
            wall_type = "top" if r <= rows // 2 else "bottom"
            break

    if wall_type is None:
        for c in range(cols):
            if all(grid[r][c] == 2 for r in range(rows)):
                wall_idx = c
                wall_type = "left" if c <= cols // 2 else "right"
                break

    # Find connected objects via BFS
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 2) and not visited[r][c]:
                color = grid[r][c]
                cells = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and not visited[nr][nc]
                            and grid[nr][nc] == color
                        ):
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                objects.append((color, cells))

    # Sort objects by distance to wall (closest first)
    def sort_key(obj):
        _, cells = obj
        if wall_type == "top":
            return min(r for r, c in cells)
        elif wall_type == "bottom":
            return -max(r for r, c in cells)
        elif wall_type == "left":
            return min(c for r, c in cells)
        else:
            return -max(c for r, c in cells)

    objects.sort(key=sort_key)

    # Create output grid with only the wall
    out = [[0] * cols for _ in range(rows)]
    if wall_type in ("top", "bottom"):
        for c in range(cols):
            out[wall_idx][c] = 2
    else:
        for r in range(rows):
            out[r][wall_idx] = 2

    # Place each object as close to the wall as possible
    for color, cells in objects:
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        if wall_type == "top":
            for target in range(wall_idx + 1, rows):
                offset = target - min_r
                if all(
                    0 <= cr + offset < rows and out[cr + offset][cc] == 0
                    for cr, cc in cells
                ):
                    for cr, cc in cells:
                        out[cr + offset][cc] = color
                    break

        elif wall_type == "bottom":
            for target in range(wall_idx - 1, -1, -1):
                offset = target - max_r
                if all(
                    0 <= cr + offset < rows and out[cr + offset][cc] == 0
                    for cr, cc in cells
                ):
                    for cr, cc in cells:
                        out[cr + offset][cc] = color
                    break

        elif wall_type == "left":
            for target in range(wall_idx + 1, cols):
                offset = target - min_c
                if all(
                    0 <= cc + offset < cols and out[cr][cc + offset] == 0
                    for cr, cc in cells
                ):
                    for cr, cc in cells:
                        out[cr][cc + offset] = color
                    break

        elif wall_type == "right":
            for target in range(wall_idx - 1, -1, -1):
                offset = target - max_c
                if all(
                    0 <= cc + offset < cols and out[cr][cc + offset] == 0
                    for cr, cc in cells
                ):
                    for cr, cc in cells:
                        out[cr][cc + offset] = color
                    break

    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/6ad5bdfd.json") as f:
        task = json.load(f)

    all_pass = True
    for split in ("train", "test"):
        for i, example in enumerate(task[split]):
            result = solve(example["input"])
            expected = example["output"]
            status = "PASS" if result == expected else "FAIL"
            print(f"{split}[{i}]: {status}")
            if status == "FAIL":
                all_pass = False
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
