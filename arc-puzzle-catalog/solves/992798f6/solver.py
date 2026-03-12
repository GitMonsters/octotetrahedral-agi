"""Solver for ARC puzzle 992798f6.

Two dots (1 and 2) on a black grid. Draw a path of 3s from dot 2 toward dot 1.
The path starts one step diagonally from 2 toward 1, ends one step diagonally
from 1 toward 2. It first takes the straight (excess) segment, then diagonal.
"""

from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Find positions of dot 1 and dot 2
    pos1 = pos2 = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                pos1 = (r, c)
            elif grid[r][c] == 2:
                pos2 = (r, c)

    def sign(x: int) -> int:
        return (x > 0) - (x < 0)

    # Path goes from adjacent-to-2 toward adjacent-to-1
    dr = sign(pos1[0] - pos2[0])
    dc = sign(pos1[1] - pos2[1])

    start = (pos2[0] + dr, pos2[1] + dc)
    end = (pos1[0] - dr, pos1[1] - dc)

    total_dr = end[0] - start[0]
    total_dc = end[1] - start[1]
    abs_dr = abs(total_dr)
    abs_dc = abs(total_dc)
    sdr = sign(total_dr)
    sdc = sign(total_dc)

    diag_steps = min(abs_dr, abs_dc)
    straight_steps = abs(abs_dr - abs_dc)

    r, c = start
    out[r][c] = 3

    # Straight segment first (excess dimension)
    if abs_dr > abs_dc:
        for _ in range(straight_steps):
            r += sdr
            out[r][c] = 3
    elif abs_dc > abs_dr:
        for _ in range(straight_steps):
            c += sdc
            out[r][c] = 3

    # Then diagonal segment
    for _ in range(diag_steps):
        r += sdr
        c += sdc
        out[r][c] = 3

    return out
