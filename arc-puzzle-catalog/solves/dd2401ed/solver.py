"""
dd2401ed solver

Rule:
1. Find the vertical wall of 5s at column W.
2. The wall moves to column W' = 2*W + 1.
3. The "zone" between old and new wall is cols W+1..2W.
4. Check conversion condition: for every left-side column that contains a 1,
   its mirror column (2*W - c) in the zone must contain at least one 2.
   If ALL such mirrors exist, convert ALL zone 2s to 1s.
5. Output = input with wall moved and zone 2s optionally converted.
"""
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    R = len(grid)
    C = len(grid[0])

    # Find wall column (all 5s)
    W = next(c for c in range(C) if all(grid[r][c] == 5 for r in range(R)))
    W2 = 2 * W + 1  # new wall position

    # Columns on the left side that contain at least one 1
    left_cols_with_1 = set()
    for r in range(R):
        for c in range(W):
            if grid[r][c] == 1:
                left_cols_with_1.add(c)

    # Mirror those columns into the zone
    mirror_cols = {2 * W - c for c in left_cols_with_1}

    # Zone columns that contain at least one 2
    zone_cols_with_2 = set()
    for r in range(R):
        for c in range(W + 1, 2 * W + 1):
            if grid[r][c] == 2:
                zone_cols_with_2.add(c)

    # Conversion happens iff every mirrored left-1-col has a matching zone-2-col
    convert = mirror_cols.issubset(zone_cols_with_2)

    # Build output
    out = [row[:] for row in grid]
    for r in range(R):
        out[r][W] = 0       # remove old wall
        out[r][W2] = 5      # place new wall
        if convert:
            for c in range(W + 1, 2 * W + 1):
                if out[r][c] == 2:
                    out[r][c] = 1

    return out


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/dd2401ed.json") as f:
        data = json.load(f)
    for phase in ["train", "test"]:
        for i, example in enumerate(data[phase]):
            result = solve(example["input"])
            expected = example["output"]
            status = "PASS" if result == expected else "FAIL"
            print(f"{phase} {i}: {status}")
            if status == "FAIL":
                for r in range(len(expected)):
                    if result[r] != expected[r]:
                        print(f"  row {r}: got {result[r]}")
                        print(f"          exp {expected[r]}")
