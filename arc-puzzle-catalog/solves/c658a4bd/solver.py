"""Solver for ARC-AGI task c658a4bd.

Pattern: The input contains several colored hollow rectangles scattered
(and partially overlapping) on a black background. The output nests them
concentrically, ordered by bounding-box size (largest = outermost layer,
smallest = innermost fill).
"""

from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])

    # Find bounding box for every non-zero color
    color_bounds: dict[int, list[int]] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v == 0:
                continue
            if v not in color_bounds:
                color_bounds[v] = [r, r, c, c]
            else:
                b = color_bounds[v]
                if r < b[0]: b[0] = r
                if r > b[1]: b[1] = r
                if c < b[2]: b[2] = c
                if c > b[3]: b[3] = c

    # Size = max(height, width) of each bounding box; sort descending
    layers = sorted(
        ((max(b[1] - b[0] + 1, b[3] - b[2] + 1), color)
         for color, b in color_bounds.items()),
        reverse=True,
    )

    n = layers[0][0]  # output side length = size of largest rectangle
    output = [[0] * n for _ in range(n)]

    # Paint from outermost to innermost (each overwrites the center)
    for i, (_size, color) in enumerate(layers):
        for r in range(i, n - i):
            for c in range(i, n - i):
                output[r][c] = color

    return output


# --------------- verification ---------------
if __name__ == "__main__":
    train = [
        {
            "input": [[0,0,0,0,0,0,0,0,0,0,0,0,0],[8,8,8,8,8,8,0,0,3,3,3,3,0],[8,0,0,0,0,8,0,0,3,0,0,3,0],[8,0,0,0,0,8,0,0,3,0,0,3,0],[8,0,0,2,2,2,2,2,3,3,3,3,0],[8,0,0,2,0,8,0,0,0,0,2,0,0],[8,8,8,2,8,8,0,0,0,0,2,0,0],[0,0,0,2,0,0,0,0,0,0,2,0,0],[0,0,0,2,0,0,0,0,0,0,2,0,0],[4,4,0,2,0,0,0,0,0,0,2,0,0],[4,4,0,2,0,0,0,0,0,0,2,0,0],[0,0,0,2,2,2,2,2,2,2,2,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0]],
            "output": [[2,2,2,2,2,2,2,2],[2,8,8,8,8,8,8,2],[2,8,3,3,3,3,8,2],[2,8,3,4,4,3,8,2],[2,8,3,4,4,3,8,2],[2,8,3,3,3,3,8,2],[2,8,8,8,8,8,8,2],[2,2,2,2,2,2,2,2]],
        },
        {
            "input": [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,4,4,4,4,4,0,0,0,0,0,0,0,0],[2,0,0,4,0,0,0,4,0,0,1,1,1,0,0,0],[0,0,0,4,0,0,0,4,0,0,1,0,1,0,0,0],[0,0,0,4,0,0,0,4,0,0,1,1,1,0,0,0],[0,0,0,4,4,4,4,4,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,8,8,8,8,8,8,8,0,0,0,0,0,0,0,0],[0,8,0,0,0,0,0,8,0,0,0,0,0,0,0,0],[0,8,0,0,0,3,3,3,3,3,3,3,3,3,0,0],[0,8,0,0,0,3,0,8,0,0,0,0,0,3,0,0],[0,8,0,0,0,3,0,8,0,0,0,0,0,3,0,0],[0,8,0,0,0,3,0,8,0,0,0,0,0,3,0,0],[0,8,8,8,8,3,8,8,0,0,0,0,0,3,0,0],[0,0,0,0,0,3,0,0,0,0,0,0,0,3,0,0],[0,0,0,0,0,3,0,0,0,0,0,0,0,3,0,0]],
            "output": [[3,3,3,3,3,3,3,3,3],[3,8,8,8,8,8,8,8,3],[3,8,4,4,4,4,4,8,3],[3,8,4,1,1,1,4,8,3],[3,8,4,1,2,1,4,8,3],[3,8,4,1,1,1,4,8,3],[3,8,4,4,4,4,4,8,3],[3,8,8,8,8,8,8,8,3],[3,3,3,3,3,3,3,3,3]],
        },
    ]

    all_pass = True
    for i, ex in enumerate(train):
        result = solve(ex["input"])
        if result == ex["output"]:
            print(f"Training example {i+1}: PASS")
        else:
            print(f"Training example {i+1}: FAIL")
            print(f"  Expected: {ex['output']}")
            print(f"  Got:      {result}")
            all_pass = False

    # Run on test input
    test_input = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,6,6,6,6,6,6,6,6,6,6,0,0,0,0,0,0,0],[0,0,6,0,0,0,0,0,0,0,0,6,0,0,0,2,2,0,0],[0,0,6,0,8,8,8,8,8,8,0,6,0,0,0,2,2,0,0],[0,0,6,0,8,0,0,0,0,8,0,6,0,0,0,0,0,0,0],[0,0,6,0,8,3,3,3,3,8,3,3,3,0,0,0,0,0,0],[0,0,6,0,8,3,0,0,0,8,0,6,3,0,0,0,0,0,0],[0,0,6,0,8,3,0,0,0,8,0,6,3,0,0,0,0,0,0],[0,0,6,0,8,8,8,8,8,8,0,6,3,0,0,0,0,0,0],[0,0,6,0,0,3,0,0,0,0,0,6,3,0,0,0,0,0,0],[0,0,6,6,6,3,6,6,6,6,6,6,3,0,0,0,0,0,0],[0,0,0,0,0,3,0,0,0,0,0,0,3,0,0,0,0,0,0],[0,0,0,0,0,3,3,3,3,3,3,3,3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,4,0,0,4,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,4,0,0,4,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    test_result = solve(test_input)
    print(f"\nTest output ({len(test_result)}x{len(test_result[0])}):")
    for row in test_result:
        print(row)

    if all_pass:
        print("\n✓ All training examples passed!")
