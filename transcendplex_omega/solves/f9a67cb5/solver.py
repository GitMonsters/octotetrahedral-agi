"""
ARC-AGI puzzle f9a67cb5 solver.

Pattern: A single '2' cell sends lines through gaps in walls of 8s.
- Walls are rows (horizontal) or columns (vertical) of 8s with gap cells (0).
- From the 2, a line propagates perpendicular to the walls.
- At each wall, a "connector" line spreads on the approach row/col to reach gaps.
- For each incoming position, the connector extends to the nearest gap in each
  direction; if no gap exists in a direction, it extends to the grid edge.
- Only gaps covered by the connector (or directly aligned with an incoming path)
  are activated. The process repeats through each wall.
- After the last wall, lines extend straight to the grid edge.
"""

import json
import copy
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Find the 2
    r2 = c2 = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                r2, c2 = r, c

    # Determine orientation by comparing max 8-count in rows vs columns
    row_8 = [sum(1 for c in range(cols) if grid[r][c] == 8) for r in range(rows)]
    col_8 = [sum(1 for r in range(rows) if grid[r][c] == 8) for c in range(cols)]

    if max(row_8) >= max(col_8):
        # Horizontal walls — propagate vertically
        threshold = cols / 2
        walls = sorted(r for r in range(rows) if row_8[r] > threshold)
        direction = 1 if r2 < walls[0] else -1
        if direction == -1:
            walls = walls[::-1]

        def gaps_of(w):
            return [c for c in range(cols) if grid[w][c] == 0]

        incoming = [c2]

        for wi, w in enumerate(walls):
            gaps = gaps_of(w)
            approach = w - direction
            start = r2 if wi == 0 else walls[wi - 1] + direction

            # Draw vertical lines from each incoming col
            if direction == 1:
                for ic in incoming:
                    for r in range(start, approach + 1):
                        result[r][ic] = 2
            else:
                for ic in incoming:
                    for r in range(start, approach - 1, -1):
                        result[r][ic] = 2

            # Compute connector spans on the approach row
            connector = set()
            direct = set()
            for ic in incoming:
                if ic in gaps:
                    direct.add(ic)
                else:
                    left = right = None
                    for g in gaps:
                        if g < ic and (left is None or g > left):
                            left = g
                        if g > ic and (right is None or g < right):
                            right = g
                    lo = left if left is not None else 0
                    hi = right if right is not None else cols - 1
                    connector.update(range(lo, hi + 1))

            for c in connector:
                result[approach][c] = 2

            activated = {g for g in gaps if g in connector or g in direct}
            for g in activated:
                result[w][g] = 2

            incoming = sorted(activated)

        # Extend past the last wall to the grid edge
        if walls:
            start = walls[-1] + direction
            rng = range(start, rows) if direction == 1 else range(start, -1, -1)
            for r in rng:
                for ic in incoming:
                    result[r][ic] = 2

    else:
        # Vertical walls — propagate horizontally
        threshold = rows / 2
        walls = sorted(c for c in range(cols) if col_8[c] > threshold)
        direction = 1 if c2 < walls[0] else -1
        if direction == -1:
            walls = walls[::-1]

        def gaps_of(w):
            return [r for r in range(rows) if grid[r][w] == 0]

        incoming = [r2]

        for wi, w in enumerate(walls):
            gaps = gaps_of(w)
            approach = w - direction
            start = c2 if wi == 0 else walls[wi - 1] + direction

            if direction == 1:
                for ir in incoming:
                    for c in range(start, approach + 1):
                        result[ir][c] = 2
            else:
                for ir in incoming:
                    for c in range(start, approach - 1, -1):
                        result[ir][c] = 2

            connector = set()
            direct = set()
            for ir in incoming:
                if ir in gaps:
                    direct.add(ir)
                else:
                    up = down = None
                    for g in gaps:
                        if g < ir and (up is None or g > up):
                            up = g
                        if g > ir and (down is None or g < down):
                            down = g
                    lo = up if up is not None else 0
                    hi = down if down is not None else rows - 1
                    connector.update(range(lo, hi + 1))

            for r in connector:
                result[r][approach] = 2

            activated = {g for g in gaps if g in connector or g in direct}
            for g in activated:
                result[g][w] = 2

            incoming = sorted(activated)

        if walls:
            start = walls[-1] + direction
            rng = range(start, cols) if direction == 1 else range(start, -1, -1)
            for c in rng:
                for ir in incoming:
                    result[ir][c] = 2

    return result


if __name__ == "__main__":
    with open("/tmp/arc_task_f9a67cb5.json") as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        out = solve(ex["input"])
        if out == ex["output"]:
            print(f"Training example {i}: PASS")
        else:
            all_pass = False
            print(f"Training example {i}: FAIL")
            for r in range(len(out)):
                if out[r] != ex["output"][r]:
                    print(f"  Row {r}: got    {out[r]}")
                    print(f"          expect {ex['output'][r]}")

    if all_pass:
        print("\nAll training examples passed!")

    # Also run on test input(s)
    for i, t in enumerate(task.get("test", [])):
        out = solve(t["input"])
        print(f"\nTest {i} output:")
        for row in out:
            print(row)
        if "output" in t:
            if out == t["output"]:
                print(f"Test {i}: PASS")
            else:
                print(f"Test {i}: FAIL")
