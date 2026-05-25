"""
ARC-AGI Puzzle 712bf12e Solver

Rule: Each 2 in the bottom row emits a vertical line upward. When the line
hits a 5, it shifts one column to the right (placing a corner piece in the
row below at the new column). If the corner position contains a 5 or is out
of bounds, the line terminates. Multiple consecutive 5s cause multiple shifts.
"""

import json
import copy
import sys


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)

    bottom_row = rows - 1
    starts = [c for c in range(cols) if grid[bottom_row][c] == 2]

    for start_col in starts:
        col = start_col
        for row in range(bottom_row - 1, -1, -1):
            placed = False
            while col < cols:
                if grid[row][col] == 5:
                    # Shift right: place corner one row below at col+1
                    new_col = col + 1
                    if new_col >= cols:
                        break
                    if grid[row + 1][new_col] == 5:
                        break
                    output[row + 1][new_col] = 2
                    col = new_col
                else:
                    output[row][col] = 2
                    placed = True
                    break
            if not placed:
                break

    return output


if __name__ == "__main__":
    task_path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/712bf12e.json"
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for i, example in enumerate(task["train"]):
        result = solve(example["input"])
        expected = example["output"]
        if result == expected:
            print(f"Training example {i}: PASS")
        else:
            print(f"Training example {i}: FAIL")
            all_pass = False
            for r in range(len(expected)):
                if result[r] != expected[r]:
                    print(f"  Row {r}: expected {expected[r]}")
                    print(f"  Row {r}: got      {result[r]}")

    for i, example in enumerate(task.get("test", [])):
        result = solve(example["input"])
        if "output" in example:
            if result == example["output"]:
                print(f"Test example {i}: PASS")
            else:
                print(f"Test example {i}: FAIL")
                all_pass = False
        else:
            print(f"Test example {i}: predicted output:")
            for row in result:
                print(f"  {row}")

    if all_pass:
        print("\nAll examples PASSED!")
    else:
        print("\nSome examples FAILED!")
        sys.exit(1)
