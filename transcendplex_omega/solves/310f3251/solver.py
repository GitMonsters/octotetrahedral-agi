"""
Solver for ARC task 310f3251.

For each non-zero cell at (r,c) in the N×N input, place a 2 at the
diagonally shifted position ((r-1)%N, (c-1)%N) if that cell is empty.
Then tile the modified grid 3×3 to produce the output.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    n = len(grid)
    tile = [row[:] for row in grid]

    for r in range(n):
        for c in range(n):
            if grid[r][c] != 0:
                sr, sc = (r - 1) % n, (c - 1) % n
                if tile[sr][sc] == 0:
                    tile[sr][sc] = 2

    # Tile 3×3
    out_size = n * 3
    return [[tile[r % n][c % n] for c in range(out_size)] for r in range(out_size)]


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/310f3251.json") as f:
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
