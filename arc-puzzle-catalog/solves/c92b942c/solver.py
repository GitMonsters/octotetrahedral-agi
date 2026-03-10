"""
ARC-AGI solver for task c92b942c.

The output grid is 3× the input in each dimension. The transformation:
1. Tile the input 3×3 across the output.
2. Every row that contains a colored (non-zero) cell in the tiled input
   is filled with 1s, with the actual color values placed at their positions.
3. On non-color rows, a 3 is placed at each cell that is diagonally adjacent
   (NW or SE, i.e. one step along the top-left → bottom-right diagonal)
   to a tiled colored cell instance.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    R = len(grid)
    C = len(grid[0])
    out_R = 3 * R
    out_C = 3 * C

    color_rows = {r for r in range(R) if any(grid[r][c] != 0 for c in range(C))}

    output = [[0] * out_C for _ in range(out_R)]

    for i in range(out_R):
        ri = i % R
        for j in range(out_C):
            ci = j % C
            if grid[ri][ci] != 0:
                output[i][j] = grid[ri][ci]
            elif ri in color_rows:
                output[i][j] = 1
            else:
                # Check if NW or SE neighbor is a tiled colored cell
                has_3 = False
                if i + 1 < out_R and j + 1 < out_C:
                    if grid[(i + 1) % R][(j + 1) % C] != 0:
                        has_3 = True
                if i > 0 and j > 0:
                    if grid[(i - 1) % R][(j - 1) % C] != 0:
                        has_3 = True
                output[i][j] = 3 if has_3 else 0

    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/c92b942c.json") as f:
        task = json.load(f)

    for split in ("train", "test"):
        for i, ex in enumerate(task[split]):
            result = solve(ex["input"])
            status = "PASS" if result == ex["output"] else "FAIL"
            print(f"{split}[{i}]: {status}")
            if status == "FAIL":
                print(f"  expected: {ex['output']}")
                print(f"  got:      {result}")
