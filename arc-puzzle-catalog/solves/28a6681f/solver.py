"""
Solver for ARC-AGI-2 puzzle 28a6681f: Staircase Interior Fill

Rule: Blue (color 1) fills the interior gaps of nested staircase shapes.
The total number of blue cells is conserved between input and output.

Algorithm:
1. Remove all blue cells from the grid, count them (N).
2. Classify each empty (bg=0) cell in the clean grid:
   - TYPE A: nearest non-bg cell to LEFT and RIGHT are the SAME color
     (these are "closed gaps" between same-color staircase walls)
   - TYPE B: has a non-bg cell to the LEFT but not a same-color match on RIGHT
     (these are "open side" extensions of the staircase)
3. Fill ALL Type A cells with blue (from bottom to top).
4. Fill Type B cells from bottom to top until total blue count = N.

Training accuracy: 3/3 (100%)
"""

import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    bg = 0
    blue = 1

    # Remove blue, count original
    clean = [[bg if grid[r][c] == blue else grid[r][c] for c in range(cols)] for r in range(rows)]
    N = sum(1 for r in range(rows) for c in range(cols) if grid[r][c] == blue)

    raw_type_a: set[tuple[int, int]] = set()
    type_b: list[tuple[int, int]] = []

    for r in range(rows):
        for c in range(cols):
            if clean[r][c] != bg:
                continue
            # Nearest non-bg to left
            l_color = None
            for cc in range(c - 1, -1, -1):
                if clean[r][cc] != bg:
                    l_color = clean[r][cc]
                    break
            # Nearest non-bg to right
            r_color = None
            for cc in range(c + 1, cols):
                if clean[r][cc] != bg:
                    r_color = clean[r][cc]
                    break

            if l_color is not None and r_color is not None and l_color == r_color:
                raw_type_a.add((r, c))
            elif l_color is not None:
                type_b.append((r, c))

    # Filter Type A: only keep cells that "stack" on a wall or another
    # valid Type A cell directly below.  This prevents filling isolated
    # same-color horizontal gaps that are vertically disconnected from
    # the base of the staircase enclosure.
    valid_a: set[tuple[int, int]] = set()
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            if (r, c) not in raw_type_a:
                continue
            if r == rows - 1 or clean[r + 1][c] != bg or (r + 1, c) in valid_a:
                valid_a.add((r, c))

    type_a = sorted(valid_a, key=lambda x: (-x[0], x[1]))

    # Sort bottom-first (higher row index = lower on grid)
    type_b.sort(key=lambda x: (-x[0], x[1]))

    result = [row[:] for row in clean]
    filled = 0

    # Fill all TYPE A (closed same-color gaps)
    for r, c in type_a:
        if filled >= N:
            break
        result[r][c] = blue
        filled += 1

    # Fill TYPE B (open extensions) from bottom until quota met
    for r, c in type_b:
        if filled >= N:
            break
        result[r][c] = blue
        filled += 1

    return result


if __name__ == "__main__":
    import os

    task_path = os.path.expanduser(
        "~/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/28a6681f.json"
    )
    with open(task_path) as f:
        task = json.load(f)

    print("=== Training Verification ===")
    for ti, ex in enumerate(task["train"]):
        pred = solve(ex["input"])
        out = ex["output"]
        rows, cols = len(out), len(out[0])
        correct = sum(
            1 for r in range(rows) for c in range(cols) if pred[r][c] == out[r][c]
        )
        total = rows * cols
        print(f"Train {ti}: {correct}/{total} {'✓' if correct == total else '✗'}")

    print("\n=== Test Predictions ===")
    for ti, test in enumerate(task["test"]):
        pred = solve(test["input"])
        print(f"Test {ti}: {len(pred)}x{len(pred[0])}")
        for row in pred:
            print("  ", row)
