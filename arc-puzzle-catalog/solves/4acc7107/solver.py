"""
Task 4acc7107 solver.

Rule: The grid contains exactly two non-background colors, each with exactly two
disconnected shapes. The output rearranges them side-by-side at the bottom:
  - Left color = color whose leftmost pixel is furthest left.
  - Right color = the other one.
  - Right side anchors at column (max_width_of_left_shapes + 1).
  - Within each side, the smaller shape (by cell count, then height) goes above,
    the larger goes below, both gravity-dropped to the last row with one blank
    separator row between them.
  - The two upper shapes share the same bottom row; the two lower shapes both
    end at the last row.
  - Each shape's relative cell arrangement is preserved from the input.
"""

from collections import Counter
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])

    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    colors = list(set(flat) - {bg})

    visited: set = set()

    def bfs(sr: int, sc: int, color: int) -> List[tuple]:
        q, cells = [(sr, sc)], []
        visited.add((sr, sc))
        while q:
            r, c = q.pop(0)
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr][nc] == color
                ):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return cells

    def get_components(color: int) -> List[List[tuple]]:
        comps = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r, c) not in visited:
                    comps.append(bfs(r, c, color))
        return comps

    def shape_height(cells: List[tuple]) -> int:
        return max(r for r, c in cells) - min(r for r, c in cells) + 1

    def shape_width(cells: List[tuple]) -> int:
        return max(c for r, c in cells) - min(c for r, c in cells) + 1

    def sort_key(cells: List[tuple]) -> tuple:
        return (len(cells), shape_height(cells))

    def leftmost_col(color: int) -> int:
        return min(c for comp in get_components(color) for _, c in comp)

    left_color = min(colors, key=leftmost_col)
    right_color = next(c for c in colors if c != left_color)

    visited.clear()
    left_comps = get_components(left_color)
    right_comps = get_components(right_color)

    left_comps.sort(key=sort_key)
    right_comps.sort(key=sort_key)

    left_small, left_large = left_comps[0], left_comps[1]
    right_small, right_large = right_comps[0], right_comps[1]

    max_left_w = max(shape_width(c) for c in left_comps)
    right_col0 = max_left_w + 1

    R = rows - 1
    h_ll = shape_height(left_large)

    left_large_bottom = R
    left_small_bottom = R - h_ll - 1
    right_large_bottom = R
    right_small_bottom = left_small_bottom

    def place_shape(
        out: List[List[int]],
        cells: List[tuple],
        color: int,
        bottom_row: int,
        col0: int,
    ) -> None:
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        for r, c in cells:
            out_r = bottom_row - (max_r - r)
            out_c = col0 + (c - min_c)
            out[out_r][out_c] = color

    out = [[bg] * cols for _ in range(rows)]
    place_shape(out, left_large, left_color, left_large_bottom, 0)
    place_shape(out, left_small, left_color, left_small_bottom, 0)
    place_shape(out, right_large, right_color, right_large_bottom, right_col0)
    place_shape(out, right_small, right_color, right_small_bottom, right_col0)

    return out
