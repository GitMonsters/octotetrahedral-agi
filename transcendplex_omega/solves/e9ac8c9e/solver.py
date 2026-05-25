"""Solver for ARC-AGI puzzle e9ac8c9e.

Pattern: Each 5-block rectangle has 4 colored corner markers diagonally adjacent
to its corners. Replace each quadrant of the 5-block with the corresponding
corner color, then erase the corner markers.
"""

import json
import copy


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = copy.deepcopy(grid)

    # Find all connected components of 5s (rectangular blocks)
    visited = [[False] * cols for _ in range(rows)]
    blocks = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and not visited[r][c]:
                # BFS to find bounding box
                tr, br, lc, rc = r, r, c, c
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    tr, br = min(tr, cr), max(br, cr)
                    lc, rc = min(lc, cc), max(rc, cc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 5:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                blocks.append((tr, br, lc, rc))

    for tr, br, lc, rc in blocks:
        # Corner markers are diagonally adjacent to the block corners
        tl_color = grid[tr - 1][lc - 1]
        t_r_color = grid[tr - 1][rc + 1]
        bl_color = grid[br + 1][lc - 1]
        br_color = grid[br + 1][rc + 1]

        # Erase corner markers
        out[tr - 1][lc - 1] = 0
        out[tr - 1][rc + 1] = 0
        out[br + 1][lc - 1] = 0
        out[br + 1][rc + 1] = 0

        h = br - tr + 1
        w = rc - lc + 1
        mid_r = h // 2
        mid_c = w // 2

        for r in range(h):
            for c in range(w):
                if r < mid_r:
                    color = tl_color if c < mid_c else t_r_color
                else:
                    color = bl_color if c < mid_c else br_color
                out[tr + r][lc + c] = color

    return out


if __name__ == "__main__":
    with open("/tmp/arc_task_e9ac8c9e.json") as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        match = result == ex["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r, (got, exp) in enumerate(zip(result, ex["output"])):
                if got != exp:
                    print(f"  Row {r}: got {got} expected {exp}")

    if "output" in task["test"][0]:
        result = solve(task["test"][0]["input"])
        match = result == task["test"][0]["output"]
        print(f"Test 0: {'PASS' if match else 'FAIL'}")

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
