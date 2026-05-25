"""
Solver for ARC-AGI puzzle 281123b4.

Pattern:
- Input is 4x19: four 4x4 blocks separated by columns of 3s (at cols 4, 9, 14).
- Block colors are always: B1=8, B2=5, B3=9, B4=4 (non-zero cells).
- Output is 4x4. For each cell, check which blocks are "active" (non-zero).
- Pick the active block's color using fixed priority: 9 > 4 > 8 > 5.
- If no block is active, output 0.
"""

import json
from pathlib import Path


# Priority order: 9 > 4 > 8 > 5
PRIORITY = {9: 0, 4: 1, 8: 2, 5: 3}


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    # Extract the four 4x4 blocks (cols 0-3, 5-8, 10-13, 15-18)
    block_starts = [0, 5, 10, 15]
    blocks = []
    for bs in block_starts:
        block = []
        for r in range(rows):
            block.append(grid[r][bs:bs+4])
        blocks.append(block)

    # Build output: for each cell, pick the highest-priority active color
    output = []
    for r in range(rows):
        row = []
        for c in range(4):
            candidates = []
            for b in blocks:
                val = b[r][c]
                if val != 0 and val != 3:
                    candidates.append(val)
            if not candidates:
                row.append(0)
            else:
                # Pick by priority
                row.append(min(candidates, key=lambda v: PRIORITY[v]))
        output.append(row)
    return output


if __name__ == "__main__":
    data_path = Path.home() / "ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/281123b4.json"
    with open(data_path) as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            all_pass = False
            print(f"Train {i}: FAIL")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

    if all_pass:
        print("\nAll training examples passed!")
        # Solve test
        for i, ex in enumerate(data["test"]):
            result = solve(ex["input"])
            print(f"\nTest {i} solution:")
            for row in result:
                print(" ".join(str(v) for v in row))
    else:
        print("\nSome examples failed!")
