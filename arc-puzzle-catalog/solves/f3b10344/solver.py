"""Solver for ARC-AGI puzzle f3b10344.

Rule: Find pairs of same-colored rectangular blocks. For adjacent pairs
(no same-colored block between them), fill the rectangular gap between
their interiors (border removed) with color 8.
"""

from typing import List
from collections import defaultdict
import copy


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)

    # Find all rectangular blocks via connected components
    visited = [[False] * cols for _ in range(rows)]
    blocks: list[tuple[int, int, int, int, int]] = []  # (color, r_min, r_max, c_min, c_max)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                queue = [(r, c)]
                visited[r][c] = True
                cells = [(r, c)]
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            cells.append((nr, nc))
                min_r = min(p[0] for p in cells)
                max_r = max(p[0] for p in cells)
                min_c = min(p[1] for p in cells)
                max_c = max(p[1] for p in cells)
                blocks.append((color, min_r, max_r, min_c, max_c))

    # Group blocks by color
    color_blocks: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for block in blocks:
        color_blocks[block[0]].append(block)

    # For each pair of same-colored blocks, try to fill the gap
    for color, blist in color_blocks.items():
        for i in range(len(blist)):
            for j in range(i + 1, len(blist)):
                _, r1_min, r1_max, c1_min, c1_max = blist[i]
                _, r2_min, r2_max, c2_min, c2_max = blist[j]

                # Interior of each block (remove 1-cell border)
                i1_r_min, i1_r_max = r1_min + 1, r1_max - 1
                i1_c_min, i1_c_max = c1_min + 1, c1_max - 1
                i2_r_min, i2_r_max = r2_min + 1, r2_max - 1
                i2_c_min, i2_c_max = c2_min + 1, c2_max - 1

                # Skip blocks too small to have an interior
                if i1_r_min > i1_r_max or i1_c_min > i1_c_max:
                    continue
                if i2_r_min > i2_r_max or i2_c_min > i2_c_max:
                    continue

                # Try horizontal pair (overlapping interior rows, separated in cols)
                ir_min = max(i1_r_min, i2_r_min)
                ir_max = min(i1_r_max, i2_r_max)
                if ir_min <= ir_max:
                    if c1_max < c2_min:
                        gap_c_min, gap_c_max = c1_max + 1, c2_min - 1
                    elif c2_max < c1_min:
                        gap_c_min, gap_c_max = c2_max + 1, c1_min - 1
                    else:
                        gap_c_min, gap_c_max = 1, 0  # overlapping, no gap

                    if gap_c_min <= gap_c_max:
                        blocked = False
                        for k in range(len(blist)):
                            if k == i or k == j:
                                continue
                            _, br_min, br_max, bc_min, bc_max = blist[k]
                            if (br_min <= ir_max and br_max >= ir_min and
                                    bc_min <= gap_c_max and bc_max >= gap_c_min):
                                blocked = True
                                break
                        if not blocked:
                            for r in range(ir_min, ir_max + 1):
                                for c in range(gap_c_min, gap_c_max + 1):
                                    if output[r][c] == 0:
                                        output[r][c] = 8

                # Try vertical pair (overlapping interior cols, separated in rows)
                ic_min = max(i1_c_min, i2_c_min)
                ic_max = min(i1_c_max, i2_c_max)
                if ic_min <= ic_max:
                    if r1_max < r2_min:
                        gap_r_min, gap_r_max = r1_max + 1, r2_min - 1
                    elif r2_max < r1_min:
                        gap_r_min, gap_r_max = r2_max + 1, r1_min - 1
                    else:
                        gap_r_min, gap_r_max = 1, 0  # overlapping, no gap

                    if gap_r_min <= gap_r_max:
                        blocked = False
                        for k in range(len(blist)):
                            if k == i or k == j:
                                continue
                            _, br_min, br_max, bc_min, bc_max = blist[k]
                            # For vertical pairs, any same-colored block whose
                            # row range overlaps the gap rows blocks the pair
                            if br_min <= gap_r_max and br_max >= gap_r_min:
                                blocked = True
                                break
                        if not blocked:
                            for r in range(gap_r_min, gap_r_max + 1):
                                for c in range(ic_min, ic_max + 1):
                                    if output[r][c] == 0:
                                        output[r][c] = 8

    return output
