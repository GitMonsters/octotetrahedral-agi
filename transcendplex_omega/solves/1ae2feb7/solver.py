"""
Solver for 1ae2feb7 — Wall frequency projection

A vertical wall divides the grid. Colored bars touch the wall on one side.
Each color group of width N repeats every N cells on the opposite side.
When colors overlap, the group closer to the wall takes priority.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(input_grid: Grid) -> Grid:
    grid = [row[:] for row in input_grid]
    H = len(grid)
    W = len(grid[0])

    # Find the vertical wall: column with the most consistent non-zero color
    wall_col = -1
    wall_color = -1
    best_count = 0
    for c in range(W):
        from collections import Counter
        col_vals = [grid[r][c] for r in range(H)]
        counts = Counter(v for v in col_vals if v != 0)
        if counts:
            color, count = counts.most_common(1)[0]
            if count > best_count:
                best_count = count
                wall_col = c
                wall_color = color

    if wall_col == -1:
        return grid

    for r in range(H):
        # Collect non-black, non-wall color groups on each side
        left_colors = {}   # color -> list of columns
        right_colors = {}
        for c in range(W):
            if c == wall_col:
                continue
            v = grid[r][c]
            if v == 0:
                continue
            if c < wall_col:
                left_colors.setdefault(v, []).append(c)
            else:
                right_colors.setdefault(v, []).append(c)

        # Determine which side has the pattern and which is the target
        if left_colors and not right_colors:
            pattern_side = left_colors
            target_range = range(wall_col + 1, W)
            # Distance from wall = wall_col - max(cols) for each group
            get_dist = lambda cols: wall_col - max(cols)
        elif right_colors and not left_colors:
            pattern_side = right_colors
            target_range = range(wall_col - 1, -1, -1)
            get_dist = lambda cols: min(cols) - wall_col
        else:
            continue

        # Build color groups with their width N and distance from wall
        groups = []
        for color, cols in pattern_side.items():
            n = len(cols)
            dist = get_dist(cols)
            groups.append((dist, color, n))

        # Sort by distance descending (farther paints first, closer overrides)
        groups.sort(key=lambda x: -x[0])

        # Paint each group onto the target side
        for _, color, n in groups:
            for i, c in enumerate(target_range):
                pos = i + 1  # 1-indexed distance from wall
                if pos % n == 1 or n == 1:
                    grid[r][c] = color

    return grid


def validate(puzzle_data: dict) -> bool:
    """Validate solver against all train pairs."""
    all_correct = True
    for i, pair in enumerate(puzzle_data["train"]):
        result = solve(pair["input"])
        expected = pair["output"]
        if result == expected:
            print(f"  Train {i}: ✅ PASS")
        else:
            print(f"  Train {i}: ❌ FAIL")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"    ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
            all_correct = False
    return all_correct


if __name__ == "__main__":
    with open("../../dataset/tasks/1ae2feb7.json") as f:
        puzzle = json.load(f)

    print("Validating solver against training examples...")
    if validate(puzzle):
        print("\n✅ All training examples pass!")
        print("\nGenerating test outputs...")
        for i, pair in enumerate(puzzle["test"]):
            result = solve(pair["input"])
            print(f"\nTest {i} output:")
            for row in result:
                print(row)
    else:
        print("\n❌ Some training examples failed.")
