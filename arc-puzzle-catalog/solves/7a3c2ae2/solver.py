from collections import Counter
from typing import List


def transform(grid: List[List[int]]) -> List[List[int]]:
    H = len(grid)
    W = len(grid[0])

    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]
    shape_c = [c for c in counts if c != bg][0]

    result = [row[:] for row in grid]

    for r1 in range(H):
        c = 0
        while c < W:
            if grid[r1][c] != shape_c:
                c += 1
                continue

            # Find horizontal run: c_start .. c_end (inclusive)
            c_start = c
            while c < W and grid[r1][c] == shape_c:
                c += 1
            c_end = c - 1

            # Need width >= 3 for a non-trivial interior
            if c_end - c_start < 2:
                continue

            # Valid top bar: walls must NOT extend above the bar at its endpoints
            if r1 > 0 and (
                grid[r1 - 1][c_start] == shape_c
                or grid[r1 - 1][c_end] == shape_c
            ):
                continue

            # Find how far left and right walls descend
            r_left = r1
            while r_left + 1 < H and grid[r_left + 1][c_start] == shape_c:
                r_left += 1

            r_right = r1
            while r_right + 1 < H and grid[r_right + 1][c_end] == shape_c:
                r_right += 1

            # Walls must extend at least one row downward
            if r_left == r1 or r_right == r1:
                continue

            r_bottom = min(r_left, r_right)

            # Case 1: closed rectangle — bottom bar present at r_bottom
            has_bottom_bar = all(
                grid[r_bottom][cc] == shape_c for cc in range(c_start, c_end + 1)
            )
            if has_bottom_bar:
                for r in range(r1 + 1, r_bottom):
                    for cc in range(c_start + 1, c_end):
                        if result[r][cc] == bg:
                            result[r][cc] = 8
                continue

            # Case 2: open-bottom frame — both walls reach the grid edge
            if r_left == H - 1 and r_right == H - 1:
                for r in range(r1 + 1, H):
                    for cc in range(c_start + 1, c_end):
                        if result[r][cc] == bg:
                            result[r][cc] = 8

    return result
