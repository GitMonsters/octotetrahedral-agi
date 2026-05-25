"""
ARC-AGI puzzle ea9794b1 solver.

Rule: The 10x10 input is four 5x5 quadrants (TL=4, TR=3, BL=9, BR=8 with 0=empty).
At each position, pick the non-zero value with priority 3 > 9 > 8 > 4.
If all zero, output 0.
"""

import json
from typing import List

PRIORITY = {3: 1, 9: 2, 8: 3, 4: 4}


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid) // 2
    cols = len(grid[0]) // 2
    output = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            candidates = [
                grid[r][c],          # TL
                grid[r][c + cols],   # TR
                grid[r + rows][c],   # BL
                grid[r + rows][c + cols],  # BR
            ]
            nonzero = [v for v in candidates if v != 0]
            if nonzero:
                output[r][c] = min(nonzero, key=lambda x: PRIORITY.get(x, 99))
    return output


if __name__ == "__main__":
    task_path = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/ea9794b1.json"
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        pred = solve(ex["input"])
        match = pred == ex["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(pred)):
                for c in range(len(pred[0])):
                    if pred[r][c] != ex["output"][r][c]:
                        print(f"  ({r},{c}): got {pred[r][c]}, expected {ex['output'][r][c]}")

    # Run on test input
    for i, ex in enumerate(task["test"]):
        pred = solve(ex["input"])
        print(f"\nTest {i} prediction:")
        for row in pred:
            print(row)
        if "output" in ex:
            match = pred == ex["output"]
            print(f"Test {i}: {'PASS' if match else 'FAIL'}")

    print(f"\nAll training examples: {'PASS' if all_pass else 'FAIL'}")
