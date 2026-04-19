"""ARC puzzle 396d80d7 solver.

Rule: There are two non-bg colors — an "outer" color (ring/frame) and an "inner" color (center).
A background cell becomes the inner color if:
  1. It is diagonally adjacent to at least one outer-color cell
  2. It is NOT 4-directionally adjacent to any non-background cell
Inner vs outer determined by mean distance from centroid (closer = inner).
"""

import copy
from collections import Counter


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])

    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    nonbg_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid[r][c] != bg]
    color_counts = Counter(input_grid[r][c] for r, c in nonbg_cells)
    colors = list(color_counts.keys())
    if len(colors) < 2:
        return copy.deepcopy(input_grid)

    centroid_r = sum(r for r, c in nonbg_cells) / len(nonbg_cells)
    centroid_c = sum(c for r, c in nonbg_cells) / len(nonbg_cells)

    def mean_dist(color: int) -> float:
        cells = [(r, c) for r, c in nonbg_cells if input_grid[r][c] == color]
        return sum(abs(r - centroid_r) + abs(c - centroid_c) for r, c in cells) / len(cells)

    if mean_dist(colors[0]) < mean_dist(colors[1]):
        inner_color, outer_color = colors[0], colors[1]
    else:
        inner_color, outer_color = colors[1], colors[0]

    output = copy.deepcopy(input_grid)

    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                continue

            has_diag_outer = any(
                0 <= r + dr < rows and 0 <= c + dc < cols
                and input_grid[r + dr][c + dc] == outer_color
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            )
            if not has_diag_outer:
                continue

            has_4adj_nonbg = any(
                0 <= r + dr < rows and 0 <= c + dc < cols
                and input_grid[r + dr][c + dc] != bg
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            if not has_4adj_nonbg:
                output[r][c] = inner_color

    return output


# === Test against all examples ===
if __name__ == "__main__":
    examples = [
        {
            "input": [[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,6,6,7,6,6,7,7,7,7,7,7],[7,7,7,7,6,7,7,6,7,7,6,7,7,7,7,7],[7,7,7,7,7,6,6,2,6,6,7,7,7,7,7,7],[7,7,7,7,6,7,7,6,7,7,6,7,7,7,7,7],[7,7,7,7,7,6,6,2,6,6,7,7,7,7,7,7],[7,7,7,7,6,7,7,6,7,7,6,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]],
            "output": [[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,2,7,7,2,7,7,2,7,7,7,7,7],[7,7,7,2,7,6,6,7,6,6,7,2,7,7,7,7],[7,7,7,7,6,7,7,6,7,7,6,7,7,7,7,7],[7,7,7,2,7,6,6,2,6,6,7,2,7,7,7,7],[7,7,7,7,6,7,7,6,7,7,6,7,7,7,7,7],[7,7,7,2,7,6,6,2,6,6,7,2,7,7,7,7],[7,7,7,7,6,7,7,6,7,7,6,7,7,7,7,7],[7,7,7,2,7,2,2,7,2,2,7,2,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]]
        },
        {
            "input": [[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,4,4,7,7,7,7,7,7,7],[7,7,7,7,7,7,4,7,7,4,7,7,7,7,7,7],[7,7,7,7,7,4,7,1,1,7,4,7,7,7,7,7],[7,7,7,7,7,4,7,1,1,7,4,7,7,7,7,7],[7,7,7,7,7,7,4,7,7,4,7,7,7,7,7,7],[7,7,7,7,7,7,7,4,4,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]],
            "output": [[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,1,7,7,1,7,7,7,7,7,7],[7,7,7,7,7,1,7,4,4,7,1,7,7,7,7,7],[7,7,7,7,1,7,4,7,7,4,7,1,7,7,7,7],[7,7,7,7,7,4,7,1,1,7,4,7,7,7,7,7],[7,7,7,7,7,4,7,1,1,7,4,7,7,7,7,7],[7,7,7,7,1,7,4,7,7,4,7,1,7,7,7,7],[7,7,7,7,7,1,7,4,4,7,1,7,7,7,7,7],[7,7,7,7,7,7,1,7,7,1,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]]
        },
    ]

    test_input = [[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,1,7,7,1,7,7,7,7,7,7],[7,7,7,7,7,7,7,9,9,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,9,9,7,7,7,7,7,7,7],[7,7,7,7,7,7,1,7,7,1,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]]

    expected_output = [[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,9,7,9,9,7,9,7,7,7,7,7],[7,7,7,7,7,7,1,7,7,1,7,7,7,7,7,7],[7,7,7,7,7,9,7,9,9,7,9,7,7,7,7,7],[7,7,7,7,7,9,7,9,9,7,9,7,7,7,7,7],[7,7,7,7,7,7,1,7,7,1,7,7,7,7,7,7],[7,7,7,7,7,9,7,9,9,7,9,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]]

    all_pass = True
    for i, ex in enumerate(examples):
        result = transform(ex["input"])
        if result == ex["output"]:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            for r in range(len(result)):
                if result[r] != ex["output"][r]:
                    print(f"  Row {r}: got {result[r]}")
                    print(f"       exp {ex['output'][r]}")
            all_pass = False

    test_result = transform(test_input)
    if test_result == expected_output:
        print("Test: PASS")
    else:
        print("Test: FAIL")
        for r in range(len(test_result)):
            if test_result[r] != expected_output[r]:
                print(f"  Row {r}: got {test_result[r]}")
                print(f"       exp {expected_output[r]}")
        all_pass = False

    print("SOLVED" if all_pass else "FAILED")
