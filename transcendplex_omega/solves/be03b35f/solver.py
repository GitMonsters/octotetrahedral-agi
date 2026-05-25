"""
ARC-AGI solver for task be03b35f.

The 5×5 input grid is divided into four 2×2 quadrants by a cross of zeros
at row 2 and column 2. The bottom-right quadrant is all 2s (the "canvas").
The output is the top-left quadrant rotated 90° counter-clockwise.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    # Extract top-left 2×2 quadrant
    a, b = grid[0][0], grid[0][1]
    c, d = grid[1][0], grid[1][1]
    # Rotate 90° CCW: [[a,b],[c,d]] → [[b,d],[a,c]]
    return [[b, d], [a, c]]


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/be03b35f.json") as f:
        task = json.load(f)

    for split in ("train", "test"):
        for i, ex in enumerate(task[split]):
            result = solve(ex["input"])
            status = "PASS" if result == ex["output"] else "FAIL"
            print(f"{split}[{i}]: {status}")
            if status == "FAIL":
                print(f"  expected: {ex['output']}")
                print(f"  got:      {result}")
