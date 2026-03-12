"""
Solver for ARC-AGI puzzle 8cb8642d

Rule: Each filled rectangle (uniform color + one interior dot of a different color)
gets its interior cleared to 0, then an X pattern is drawn using the dot's color.
The X consists of 45-degree diagonal arms from all 4 interior corners, connected
at the center by a horizontal or vertical line when the interior is non-square.
"""

import copy
import json
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    output = copy.deepcopy(grid)

    visited = [[False] * W for _ in range(H)]
    rectangles: list[tuple[int, int, int, int, int, int]] = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and not visited[r][c]:
                # BFS over all non-zero cells to find connected component
                queue = deque([(r, c)])
                visited[r][c] = True
                component = [(r, c)]
                colors = {grid[r][c]}

                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            component.append((nr, nc))
                            colors.add(grid[nr][nc])

                if len(colors) != 2:
                    continue

                min_r = min(r for r, _ in component)
                max_r = max(r for r, _ in component)
                min_c = min(c for _, c in component)
                max_c = max(c for _, c in component)

                if len(component) != (max_r - min_r + 1) * (max_c - min_c + 1):
                    continue

                border_color = grid[min_r][min_c]

                valid = True
                for j in range(min_c, max_c + 1):
                    if grid[min_r][j] != border_color or grid[max_r][j] != border_color:
                        valid = False
                        break
                if valid:
                    for i in range(min_r, max_r + 1):
                        if grid[i][min_c] != border_color or grid[i][max_c] != border_color:
                            valid = False
                            break
                if not valid:
                    continue

                dot_color = None
                for i in range(min_r + 1, max_r):
                    for j in range(min_c + 1, max_c):
                        if grid[i][j] != border_color:
                            dot_color = grid[i][j]
                            break
                    if dot_color:
                        break

                if dot_color is None:
                    continue

                rectangles.append((min_r, max_r, min_c, max_c, border_color, dot_color))

    for r1, r2, c1, c2, border_color, dot_color in rectangles:
        int_h = r2 - r1 - 1
        int_w = c2 - c1 - 1

        # Clear interior
        for i in range(r1 + 1, r2):
            for j in range(c1 + 1, c2):
                output[i][j] = 0

        s = min((int_h - 1) // 2, (int_w - 1) // 2)

        # Diagonal arms from all 4 corners
        for k in range(s + 1):
            output[r1 + 1 + k][c1 + 1 + k] = dot_color
            output[r1 + 1 + k][c2 - 1 - k] = dot_color
            output[r2 - 1 - k][c1 + 1 + k] = dot_color
            output[r2 - 1 - k][c2 - 1 - k] = dot_color

        # Horizontal connecting line at center row(s)
        for j in range(s, int_w - s):
            output[r1 + 1 + s][c1 + 1 + j] = dot_color
            output[r2 - 1 - s][c1 + 1 + j] = dot_color

        # Vertical connecting line at center column(s)
        for i in range(s, int_h - s):
            output[r1 + 1 + i][c1 + 1 + s] = dot_color
            output[r1 + 1 + i][c2 - 1 - s] = dot_color

    return output


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/8cb8642d.json"))

    all_pass = True
    for idx, example in enumerate(task["train"]):
        result = solve(example["input"])
        expected = example["output"]
        if result == expected:
            print(f"Train {idx}: PASS")
        else:
            print(f"Train {idx}: FAIL")
            all_pass = False
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

    for idx, example in enumerate(task["test"]):
        result = solve(example["input"])
        if "output" in example:
            expected = example["output"]
            if result == expected:
                print(f"Test  {idx}: PASS")
            else:
                print(f"Test  {idx}: FAIL")
                all_pass = False
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        if result[r][c] != expected[r][c]:
                            print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
        else:
            print(f"Test  {idx}: (no expected output to compare)")

    print(f"\n{'ALL PASSED!' if all_pass else 'SOME FAILED'}")
