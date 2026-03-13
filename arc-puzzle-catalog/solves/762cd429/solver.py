"""
Solver for ARC-AGI task 762cd429.

Pattern: A 2x2 seed [[a,b],[c,d]] expands in a staircase fractal.
Each column group k has width 2^(k+1), starting at column 2^(k+1)-2.
Within group k, the seed is scaled by 2^k: each value becomes a 2^k × 2^k block,
arranged as [[a_block, b_block], [c_block, d_block]].
The top half extends upward from the seed row, the bottom half extends downward.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find the 2x2 seed (first non-zero cell is top-left)
    seed_r = seed_c = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                seed_r, seed_c = r, c
                break
        if seed_r >= 0:
            break

    a, b = grid[seed_r][seed_c], grid[seed_r][seed_c + 1]
    c, d = grid[seed_r + 1][seed_c], grid[seed_r + 1][seed_c + 1]

    out = [[0] * cols for _ in range(rows)]

    col_start = seed_c
    k = 0
    while col_start < cols:
        half = 1 << k        # 2^k
        width = half << 1     # 2^(k+1)

        for dr in range(half):
            tr = seed_r - half + 1 + dr   # top half row
            br = seed_r + 1 + dr          # bottom half row

            for dc in range(width):
                col = col_start + dc
                if col >= cols:
                    break
                val_top = a if dc < half else b
                val_bot = c if dc < half else d

                if 0 <= tr < rows:
                    out[tr][col] = val_top
                if 0 <= br < rows:
                    out[br][col] = val_bot

        col_start += width
        k += 1

    return out


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/762cd429.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        expected = pair["output"]
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(expected)):
                if result[r] != expected[r]:
                    print(f"  Row {r} expected: {expected[r]}")
                    print(f"  Row {r} got:      {result[r]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        print(f"\nTest {i} output:")
        for row in result:
            print(row)
        if "output" in pair:
            match = result == pair["output"]
            print(f"Test {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
