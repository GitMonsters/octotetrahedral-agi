"""
ARC-AGI puzzle f21745ec solver.

Pattern: The grid contains several colored rectangular borders. Exactly one
rectangle has a pattern drawn inside it (the "source"). All other rectangles
are either empty targets or non-matching.

- Targets whose interior dimensions match the source get the pattern copied
  in, with the source color replaced by the target's color.
- Non-matching rectangles are erased entirely.
- The source rectangle stays unchanged.
"""

import json
import copy
from typing import List, Tuple, Dict, Optional


Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Collect unique non-zero colors
    colors: set[int] = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                colors.add(grid[r][c])

    # Find bounding box for each color
    rectangles: Dict[int, Tuple[int, int, int, int]] = {}
    for color in colors:
        min_r, max_r = rows, 0
        min_c, max_c = cols, 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
        rectangles[color] = (min_r, min_c, max_r, max_c)

    # Find the source rectangle (interior has non-zero cells)
    source_color: Optional[int] = None
    source_pattern: Optional[Grid] = None
    source_dims: Optional[Tuple[int, int]] = None

    for color, (r1, c1, r2, c2) in rectangles.items():
        has_pattern = False
        interior: Grid = []
        for r in range(r1 + 1, r2):
            row = []
            for c in range(c1 + 1, c2):
                if grid[r][c] != 0:
                    has_pattern = True
                row.append(grid[r][c])
            interior.append(row)

        if has_pattern:
            source_color = color
            source_pattern = interior
            source_dims = (r2 - r1 - 1, c2 - c1 - 1)

    assert source_color is not None and source_pattern is not None

    # Process each target rectangle
    for color, (r1, c1, r2, c2) in rectangles.items():
        if color == source_color:
            continue

        int_rows = r2 - r1 - 1
        int_cols = c2 - c1 - 1

        if (int_rows, int_cols) == source_dims:
            # Copy pattern with color substitution
            for ir in range(int_rows):
                for ic in range(int_cols):
                    if source_pattern[ir][ic] == source_color:
                        result[r1 + 1 + ir][c1 + 1 + ic] = color
                    else:
                        result[r1 + 1 + ir][c1 + 1 + ic] = 0
        else:
            # Erase non-matching rectangle
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    result[r][c] = 0

    return result


def main():
    with open("/tmp/arc_task_f21745ec.json") as f:
        task = json.load(f)

    all_pass = True
    for i, example in enumerate(task["train"]):
        predicted = solve(example["input"])
        expected = example["output"]
        match = predicted == expected
        print(f"Training example {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(expected)):
                if predicted[r] != expected[r]:
                    print(f"  Row {r}: got {predicted[r]}")
                    print(f"       exp {expected[r]}")

    if all_pass:
        print("\nAll training examples passed!")

    # Run test
    for i, test in enumerate(task.get("test", [])):
        predicted = solve(test["input"])
        print(f"\nTest {i} output:")
        for row in predicted:
            print(row)
        if "output" in test:
            match = predicted == test["output"]
            print(f"Test {i}: {'PASS' if match else 'FAIL'}")


if __name__ == "__main__":
    main()
