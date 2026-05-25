"""
Solver for ARC task e6de6e8f.

The 2-row input encodes a domino tumbling path. Non-zero column groups
(separated by all-zero columns) represent steps:
  - 2-col group with left=(2,2), right=(0,2): step RIGHT (1 output row, 2 cols)
  - 2-col group with left=(0,2), right=(2,2): step LEFT  (1 output row, 2 cols)
  - 1-col group (2,2):                        STRAIGHT   (2 output rows, 1 col)

Output: 7-wide grid. Green marker (3) at (0,3). Path rendered below.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    # Parse groups of consecutive non-zero columns
    ncols = len(grid[0])
    groups = []
    i = 0
    while i < ncols:
        if grid[0][i] != 0 or grid[1][i] != 0:
            start = i
            while i < ncols and (grid[0][i] != 0 or grid[1][i] != 0):
                i += 1
            group = [(grid[0][j], grid[1][j]) for j in range(start, i)]
            groups.append(group)
        else:
            i += 1

    # Classify each group
    steps = []
    for group in groups:
        if len(group) == 1:
            steps.append("straight")
        elif len(group) == 2:
            if group[0] == (2, 2):
                steps.append("right")
            else:
                steps.append("left")

    # Build output
    out_width = 7
    marker_col = 3
    output: list[list[int]] = []

    marker_row = [0] * out_width
    marker_row[marker_col] = 3
    output.append(marker_row)

    p = marker_col
    for step in steps:
        if step == "right":
            row = [0] * out_width
            row[p] = 2
            row[p + 1] = 2
            output.append(row)
            p += 1
        elif step == "left":
            row = [0] * out_width
            row[p - 1] = 2
            row[p] = 2
            output.append(row)
            p -= 1
        else:  # straight
            for _ in range(2):
                row = [0] * out_width
                row[p] = 2
                output.append(row)

    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/e6de6e8f.json") as f:
        task = json.load(f)

    all_pass = True
    for i, example in enumerate(task["train"] + task["test"]):
        result = solve(example["input"])
        expected = example["output"]
        label = f"train[{i}]" if i < len(task["train"]) else f"test[{i - len(task['train'])}]"
        if result == expected:
            print(f"  {label}: PASS")
        else:
            all_pass = False
            print(f"  {label}: FAIL")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")

    print("ALL PASS" if all_pass else "SOME FAILED")
