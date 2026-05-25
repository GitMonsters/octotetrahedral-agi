"""
Solver for ARC task ad7e01d0.

The input is an NxN grid containing the value 5 and other values.
The output is an N*N x N*N grid composed of NxN blocks: each block (br, bc)
is a copy of the original grid if input[br][bc] == 5, otherwise all zeros.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    N = len(grid)
    size = N * N
    out = [[0] * size for _ in range(size)]

    for br in range(N):
        for bc in range(N):
            if grid[br][bc] == 5:
                for r in range(N):
                    for c in range(N):
                        out[br * N + r][bc * N + c] = grid[r][c]
    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/ad7e01d0.json") as f:
        task = json.load(f)

    for split in ("train", "test"):
        for i, example in enumerate(task[split]):
            result = solve(example["input"])
            expected = example["output"]
            status = "PASS" if result == expected else "FAIL"
            print(f"{split}[{i}]: {status}")
            if status == "FAIL":
                for r_idx, (got, exp) in enumerate(zip(result, expected)):
                    if got != exp:
                        print(f"  row {r_idx}: got {got}")
                        print(f"  row {r_idx}: exp {exp}")
