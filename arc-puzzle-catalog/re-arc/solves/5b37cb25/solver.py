"""ARC puzzle 5b37cb25 solver.

Rule: The grid has a colored frame border (4 different colors: top/bottom/left/right, black corners).
Inside the frame, shapes on a fill background have triangular notches (indentations).
Each notch is filled with the border color it points toward, and reflected into a + cross
that extends 1 cell beyond the shape edge toward that border.
"""
import json
import copy
from collections import Counter


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = copy.deepcopy(input_grid)
    rows = len(grid)
    cols = len(grid[0])

    # 1. Identify border colors from frame edges
    top_color = grid[0][1]
    bottom_color = grid[rows - 1][1]
    left_color = grid[1][0]
    right_color = grid[1][cols - 1]

    # 2. Identify fill color (most common in interior)
    interior_colors: Counter = Counter()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            interior_colors[grid[r][c]] += 1
    fill_color = interior_colors.most_common(1)[0][0]

    # 3. Collect shape colors (anything non-fill in interior)
    shape_colors: set[int] = set()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r][c] != fill_color:
                shape_colors.add(grid[r][c])

    # 4. Find notch tips: fill cells with exactly 3 shape-colored orthogonal neighbors
    directions = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]
    border_map = {"up": top_color, "down": bottom_color, "left": left_color, "right": right_color}
    changes: list[tuple[int, int, int]] = []  # (row, col, color)

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r][c] != fill_color:
                continue

            shape_nbrs = []
            fill_nbrs = []
            for dr, dc, name in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] in shape_colors:
                        shape_nbrs.append(name)
                    elif grid[nr][nc] == fill_color:
                        fill_nbrs.append((dr, dc, name))

            if len(shape_nbrs) != 3 or len(fill_nbrs) != 1:
                continue

            # Found a notch tip! Open direction is toward the fill neighbor.
            od_dr, od_dc, od_name = fill_nbrs[0]
            border_color = border_map[od_name]

            # Cross center = 1 step from tip in open direction
            cr, cc = r + od_dr, c + od_dc
            # Extension = 2 steps from tip in open direction
            er, ec = r + 2 * od_dr, c + 2 * od_dc
            # Perpendicular cells from center
            if od_dr == 0:  # horizontal open → perp is vertical
                p1r, p1c = cr - 1, cc
                p2r, p2c = cr + 1, cc
            else:  # vertical open → perp is horizontal
                p1r, p1c = cr, cc - 1
                p2r, p2c = cr, cc + 1

            cross_cells = [(r, c), (cr, cc), (er, ec), (p1r, p1c), (p2r, p2c)]

            # Verify all cross cells are valid fill cells
            if not all(
                0 <= rr < rows and 0 <= cc2 < cols and grid[rr][cc2] == fill_color
                for rr, cc2 in cross_cells
            ):
                continue

            for rr, cc2 in cross_cells:
                changes.append((rr, cc2, border_color))

    # Apply all changes
    for r, c, color in changes:
        grid[r][c] = color

    return grid


# ── Testing ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, urllib.request

    # Load puzzle data
    data_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/refs/heads/master/data/training/5b37cb25.json"
    cache_path = "/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/5b37cb25.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            puzzle = json.load(f)
    else:
        # Inline the training data from the prompt
        puzzle = None

    if puzzle:
        all_pass = True
        for i, ex in enumerate(puzzle.get("train", [])):
            inp = ex["input"]
            expected = ex["output"]
            result = transform(inp)
            match = result == expected
            all_pass = all_pass and match
            print(f"Example {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        if result[r][c] != expected[r][c]:
                            print(f"  Diff at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

        print("\nSOLVED" if all_pass else "\nFAILED")
    else:
        print("Could not load puzzle data")
