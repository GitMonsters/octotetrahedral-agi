"""
ARC-AGI Solver for task e5790162

Rule: Starting from the cell colored 3 (always in column 0), trace an
L-shaped path going RIGHT. When the path encounters a target cell (6 or 8)
in its current row/column, it stops one cell before the target and turns:
  - 6 → clockwise 90° turn
  - 8 → counterclockwise 90° turn
All cells along the path (excluding targets) are filled with 3.
If no target is found, the path continues to the grid edge.
"""

# Direction vectors: RIGHT, DOWN, LEFT, UP
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find the starting cell (value 3)
    start_r = start_c = -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                start_r, start_c = r, c
                break
        if start_r >= 0:
            break

    # Build lookup: for each row, sorted cols of targets; for each col, sorted rows of targets
    targets: dict[tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in (6, 8):
                targets[(r, c)] = grid[r][c]

    cr, cc = start_r, start_c
    di = 0  # start going RIGHT

    while True:
        dr, dc = DIRS[di]
        # Find the nearest target in the current direction
        hit_r, hit_c, hit_val = -1, -1, -1
        nr, nc = cr + dr, cc + dc
        while 0 <= nr < rows and 0 <= nc < cols:
            if (nr, nc) in targets:
                hit_r, hit_c = nr, nc
                hit_val = targets[(nr, nc)]
                break
            nr += dr
            nc += dc

        if hit_val > 0:
            # Fill from current position to one cell before the target
            stop_r, stop_c = hit_r - dr, hit_c - dc
            fr, fc = cr, cc
            while (fr, fc) != (stop_r + dr, stop_c + dc):
                result[fr][fc] = 3
                fr += dr
                fc += dc
            # Turn based on target type
            if hit_val == 6:
                di = (di + 1) % 4  # clockwise
            else:
                di = (di - 1) % 4  # counterclockwise
            cr, cc = stop_r, stop_c
        else:
            # No target found — fill to grid edge
            nr, nc = cr, cc
            while 0 <= nr < rows and 0 <= nc < cols:
                result[nr][nc] = 3
                nr += dr
                nc += dc
            break

    return result


if __name__ == "__main__":
    import json, os

    task_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "tasks", "e5790162.json")
    with open(task_path) as f:
        data = json.load(f)

    all_pass = True
    for split in ["train", "test"]:
        for i, ex in enumerate(data[split]):
            result = solve(ex["input"])
            ok = result == ex["output"]
            print(f"{split}[{i}]: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                for r in range(len(ex["output"])):
                    for c in range(len(ex["output"][0])):
                        if result[r][c] != ex["output"][r][c]:
                            print(f"  ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
