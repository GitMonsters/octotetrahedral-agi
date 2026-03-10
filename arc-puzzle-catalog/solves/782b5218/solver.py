"""
ARC-AGI solver for task 782b5218

Transformation: The grid has 0s, 2s, and one other color C. The 2s form a
wall/divider. For each column: keep 2s in place, fill everything above the
topmost 2 with 0, fill everything below the bottommost 2 with C. Columns
with no 2 become all 0.
"""
import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find the fill color C (the non-0, non-2 color)
    fill_color = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in (0, 2):
                fill_color = grid[r][c]
                break
        if fill_color != 0:
            break

    out = [[0] * cols for _ in range(rows)]

    for c in range(cols):
        # Find 2-positions in this column
        two_rows = [r for r in range(rows) if grid[r][c] == 2]

        if not two_rows:
            continue

        top_2 = min(two_rows)
        bot_2 = max(two_rows)

        for r in range(rows):
            if r < top_2:
                out[r][c] = 0
            elif r <= bot_2:
                out[r][c] = 2
            else:
                out[r][c] = fill_color

    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/782b5218.json") as f:
        task = json.load(f)

    all_pass = True
    for split in ("train", "test"):
        for i, example in enumerate(task[split]):
            result = solve(example["input"])
            expected = example["output"]
            status = "PASS" if result == expected else "FAIL"
            print(f"{split}[{i}]: {status}")
            if status == "FAIL":
                all_pass = False
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
