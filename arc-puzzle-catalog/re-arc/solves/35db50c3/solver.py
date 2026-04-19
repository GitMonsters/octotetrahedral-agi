"""Solver for ARC puzzle 35db50c3.

Rule: Each 2x2 block emits a diagonal trail of single pixels.
- Color 9 blocks: trail goes UP-LEFT from top-left corner
- Other non-background blocks: trail goes DOWN-RIGHT from bottom-right corner
Trail continues until hitting grid boundary.
"""

from collections import Counter
from typing import List

Grid = List[List[int]]


def transform(grid: Grid) -> Grid:
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    flat = [c for r in grid for c in r]
    bg = Counter(flat).most_common(1)[0][0]

    # Copy input to output
    out = [row[:] for row in grid]

    # Find all 2x2 blocks of non-background color
    occupied = set()
    blocks = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v = grid[r][c]
            if v == bg:
                continue
            if v == grid[r + 1][c] == grid[r][c + 1] == grid[r + 1][c + 1]:
                if (r, c) not in occupied:
                    blocks.append((r, c, v))
                    occupied.update([(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)])

    # Draw diagonal trails
    for br, bc, color in blocks:
        if color == 9:
            # Trail UP-LEFT from top-left corner
            r, c = br - 1, bc - 1
            while r >= 0 and c >= 0:
                out[r][c] = color
                r -= 1
                c -= 1
        else:
            # Trail DOWN-RIGHT from bottom-right corner
            r, c = br + 2, bc + 2
            while r < rows and c < cols:
                out[r][c] = color
                r += 1
                c += 1

    return out
