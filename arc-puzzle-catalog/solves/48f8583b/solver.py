"""Solver for ARC task 48f8583b.

Rule: Find the least frequent color in the 3x3 input. Create a 9x9 grid
(3x3 arrangement of 3x3 blocks). Place copies of the input at the block
positions corresponding to cells containing the least frequent color.
"""
import json
from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    n = len(grid)
    counts = Counter(val for row in grid for val in row)
    min_color = min(counts, key=counts.get)

    output = [[0] * (n * n) for _ in range(n * n)]
    for r in range(n):
        for c in range(n):
            if grid[r][c] == min_color:
                for dr in range(n):
                    for dc in range(n):
                        output[r * n + dr][c * n + dc] = grid[dr][dc]
    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/48f8583b.json") as f:
        task = json.load(f)

    for i, example in enumerate(task["train"] + task["test"]):
        result = solve(example["input"])
        expected = example["output"]
        status = "PASS" if result == expected else "FAIL"
        label = f"train[{i}]" if i < len(task["train"]) else f"test[{i - len(task['train'])}]"
        print(f"{label}: {status}")
        if status == "FAIL":
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
