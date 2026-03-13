"""
ARC-AGI Puzzle 896d5239 Solver

Pattern: The grid contains 3s forming V-shaped triangles. Each V has a vertex
where two diagonal arms meet. The interior of each triangle is filled with 8.
Cells with 3 remain as 3; all other cells inside the triangle become 8.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    grid = [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])

    threes = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                threes.add((r, c))

    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def get_arm_length(r0: int, c0: int, dr: int, dc: int, max_gap: int = 2) -> int:
        """Find farthest 3 along diagonal, stopping after max_gap consecutive non-3 cells."""
        length = 0
        gap = 0
        for k in range(1, max(rows, cols)):
            nr, nc = r0 + k * dr, c0 + k * dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                break
            if (nr, nc) in threes:
                length = k
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
        return length

    for (r0, c0) in list(threes):
        arms = {}
        for dr, dc in directions:
            al = get_arm_length(r0, c0, dr, dc)
            if al > 0:
                arms[(dr, dc)] = al

        dir_list = sorted(arms.keys())
        for i in range(len(dir_list)):
            for j in range(i + 1, len(dir_list)):
                d1, d2 = dir_list[i], dir_list[j]
                # Skip opposite directions
                if d1[0] == -d2[0] and d1[1] == -d2[1]:
                    continue

                n1, n2 = arms[d1], arms[d2]
                N = max(n1, n2)

                if d1[0] == d2[0]:  # Same row direction — fill columns
                    dr = d1[0]
                    for k in range(1, N + 1):
                        row = r0 + k * dr
                        if not (0 <= row < rows):
                            continue
                        c_left = c0 + k * min(d1[1], d2[1])
                        c_right = c0 + k * max(d1[1], d2[1])
                        c_left = max(c_left, 0)
                        c_right = min(c_right, cols - 1)
                        for c in range(c_left, c_right + 1):
                            if grid[row][c] != 3:
                                grid[row][c] = 8

                elif d1[1] == d2[1]:  # Same column direction — fill rows
                    dc = d1[1]
                    for k in range(1, N + 1):
                        col = c0 + k * dc
                        if not (0 <= col < cols):
                            continue
                        r_top = r0 + k * min(d1[0], d2[0])
                        r_bot = r0 + k * max(d1[0], d2[0])
                        r_top = max(r_top, 0)
                        r_bot = min(r_bot, rows - 1)
                        for r in range(r_top, r_bot + 1):
                            if grid[r][col] != 3:
                                grid[r][col] = 8

    return grid


if __name__ == "__main__":
    import sys

    task_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/arc_task_896d5239.json"
    with open(task_file) as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        if result == ex["output"]:
            print(f"Training example {i}: PASS")
        else:
            print(f"Training example {i}: FAIL")
            all_pass = False

    if all_pass:
        print("\nAll training examples pass!")
