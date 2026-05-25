"""Solver for ARC-AGI task 9def23fe.

Pattern: A rectangle of 2s has colored dots scattered outside it on each side.
For each side, rectangle rows/columns WITHOUT a dot extend with 2s to the grid edge.
Rows/columns WITH a dot are left unchanged (the dot acts as a "block").
"""
import json
import copy


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    out = copy.deepcopy(grid)

    # Find bounding box of the 2-rectangle
    min_r = min_c = float('inf')
    max_r = max_c = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                min_r, max_r = min(min_r, r), max(max_r, r)
                min_c, max_c = min(min_c, c), max(max_c, c)

    # Classify each colored dot by which side of the rectangle it faces
    top_cols = set()
    bottom_cols = set()
    left_rows = set()
    right_rows = set()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 2):
                if r < min_r and min_c <= c <= max_c:
                    top_cols.add(c)
                elif r > max_r and min_c <= c <= max_c:
                    bottom_cols.add(c)
                elif c < min_c and min_r <= r <= max_r:
                    left_rows.add(r)
                elif c > max_c and min_r <= r <= max_r:
                    right_rows.add(r)

    # Extend unblocked columns upward (TOP)
    for c in range(min_c, max_c + 1):
        if c not in top_cols:
            for r in range(0, min_r):
                out[r][c] = 2

    # Extend unblocked columns downward (BOTTOM)
    for c in range(min_c, max_c + 1):
        if c not in bottom_cols:
            for r in range(max_r + 1, rows):
                out[r][c] = 2

    # Extend unblocked rows leftward (LEFT)
    for r in range(min_r, max_r + 1):
        if r not in left_rows:
            for c in range(0, min_c):
                out[r][c] = 2

    # Extend unblocked rows rightward (RIGHT)
    for r in range(min_r, max_r + 1):
        if r not in right_rows:
            for c in range(max_c + 1, cols):
                out[r][c] = 2

    return out


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/9def23fe.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        match = result == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(result)):
                if result[r] != pair["output"][r]:
                    print(f"  Row {r}: got    {result[r]}")
                    print(f"          expect {pair['output'][r]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            match = result == pair["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
        else:
            print(f"Test  {i}: (no expected output) result computed")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
