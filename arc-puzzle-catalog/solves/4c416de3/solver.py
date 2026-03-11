"""
ARC-AGI puzzle 4c416de3 solver.

Pattern: Rectangles bordered by 0s contain colored "seed" pixels at interior
corners. Each seed generates two 3-cell L-shapes: one at the rectangle's
corner (extending outward) and one at the pixel (extending inward), plus
intermediate diagonal cells if the seed is more than 1 step from the corner.
"""
import json
import copy
from collections import deque


def transform(grid):
    h, w = len(grid), len(grid[0])
    bg = grid[0][0]
    result = [row[:] for row in grid]

    # Find 0-cell connected components via BFS -> rectangles
    zero_cells = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                zero_cells.add((r, c))

    visited = set()
    rectangles = []
    for start in sorted(zero_cells):
        if start in visited:
            continue
        q = deque([start])
        visited.add(start)
        group = [start]
        while q:
            cr, cc = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < h and 0 <= nc < w
                        and (nr, nc) not in visited and grid[nr][nc] == 0):
                    visited.add((nr, nc))
                    q.append((nr, nc))
                    group.append((nr, nc))
        rows = [p[0] for p in group]
        cols = [p[1] for p in group]
        rectangles.append((min(rows), min(cols), max(rows), max(cols)))

    # Corner configs: (dr_inward, dc_inward, outward_ax1, outward_ax2)
    corner_defs = {
        'TL': (+1, +1, (-1, 0), (0, -1)),
        'TR': (+1, -1, (-1, 0), (0, +1)),
        'BL': (-1, +1, (+1, 0), (0, -1)),
        'BR': (-1, -1, (+1, 0), (0, +1)),
    }

    for top, left, bottom, right in rectangles:
        int_h = bottom - top - 1
        int_w = right - left - 1
        max_scan = max(1, min(int_h, int_w) // 2 + 1)

        corners = {'TL': (top, left), 'TR': (top, right),
                   'BL': (bottom, left), 'BR': (bottom, right)}

        for cname, (cr, cc) in corners.items():
            dr_in, dc_in, (ax1r, ax1c), (ax2r, ax2c) = corner_defs[cname]

            # Scan diagonal inward from corner to find seed pixel
            seed = None
            for d in range(1, max_scan + 1):
                pr, pc = cr + d * dr_in, cc + d * dc_in
                if not (0 <= pr < h and 0 <= pc < w):
                    break
                val = grid[pr][pc]
                if val != bg and val != 0:
                    seed = (pr, pc, val, d)
                    break

            if seed is None:
                continue

            pr, pc, color, dist = seed

            # Corner L: corner cell + two outward-adjacent cells
            cells = [
                (cr, cc),
                (cr + ax1r, cc + ax1c),
                (cr + ax2r, cc + ax2c),
            ]
            # Pixel L: pixel cell + two inward-adjacent cells
            cells += [
                (pr, pc),
                (pr - ax1r, pc - ax1c),
                (pr - ax2r, pc - ax2c),
            ]
            # Intermediate diagonal cells (for distance > 1)
            for i in range(1, dist):
                cells.append((cr + i * dr_in, cc + i * dc_in))

            for r, c in cells:
                if 0 <= r < h and 0 <= c < w:
                    result[r][c] = color

    return result


if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/4c416de3.json") as f:
        puzzle = json.load(f)

    # Validate on training examples
    for i, ex in enumerate(puzzle["train"]):
        res = transform(ex["input"])
        status = "PASS" if res == ex["output"] else "FAIL"
        print(f"Training {i}: {status}")
        if status == "FAIL":
            for r in range(len(res)):
                for c in range(len(res[0])):
                    if res[r][c] != ex["output"][r][c]:
                        print(f"  ({r},{c}): got {res[r][c]}, expected {ex['output'][r][c]}")

    # Validate on test case
    test = puzzle["test"][0]
    res = transform(test["input"])
    status = "PASS" if res == test["output"] else "FAIL"
    print(f"Test: {status}")
    if status == "FAIL":
        diffs = 0
        for r in range(len(res)):
            for c in range(len(res[0])):
                if res[r][c] != test["output"][r][c]:
                    diffs += 1
                    if diffs <= 20:
                        print(f"  ({r},{c}): got {res[r][c]}, expected {test['output'][r][c]}")
        print(f"  Total diffs: {diffs}")
