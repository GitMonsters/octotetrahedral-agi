"""Solver for ARC task 8719f442.

Rule: Input is a 3x3 grid of 0s and 5s. Output is a 15x15 grid viewed as a
5x5 arrangement of 3x3 blocks:
- Inner blocks (r+1, c+1): filled with all 5s where input[r][c]==5, else empty.
- Border blocks: copy of the input pattern placed where a 5-cell touches
  the edge (top row, bottom row, left col, right col of the input).
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    n = len(grid)  # 3
    block_n = 2 * n - 1  # 5
    out_size = block_n * n  # 15
    output = [[0] * out_size for _ in range(out_size)]

    def place_block(br: int, bc: int, content: list[list[int]]) -> None:
        for dr in range(n):
            for dc in range(n):
                output[br * n + dr][bc * n + dc] = content[dr][dc]

    filled = [[5] * n for _ in range(n)]

    # Inner 3x3 of the 5x5 block grid: filled blocks where input is 5
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 5:
                place_block(r + 1, c + 1, filled)

    # Border: input copies extending outward from edge 5-cells
    for c in range(n):
        if grid[0][c] == 5:
            place_block(0, c + 1, grid)
        if grid[n - 1][c] == 5:
            place_block(block_n - 1, c + 1, grid)
    for r in range(n):
        if grid[r][0] == 5:
            place_block(r + 1, 0, grid)
        if grid[r][n - 1] == 5:
            place_block(r + 1, block_n - 1, grid)

    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/8719f442.json") as f:
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
