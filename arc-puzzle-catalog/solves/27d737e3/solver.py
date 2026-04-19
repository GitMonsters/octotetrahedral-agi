"""
Solver for ARC task 27d737e3.

Rule: Each 2x2 block of a non-background color emits a single-pixel diagonal trail
toward a grid edge. There are two non-bg color roles:
  - "up-right" color: trail goes from the top-right corner (r, c+1) diagonally
    up-right (-1,+1) until hitting the grid boundary.
  - "down-left" color: trail goes from the bottom-left corner (r+1, c) diagonally
    down-left (+1,-1) until hitting the grid boundary.

The color-to-direction mapping is learned from the training pairs.
"""

import json
from collections import Counter

def find_bg(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]

def find_blocks(grid, bg):
    H, W = len(grid), len(grid[0])
    blocks = []
    visited = set()
    for r in range(H - 1):
        for c in range(W - 1):
            if (r, c) not in visited and grid[r][c] != bg:
                color = grid[r][c]
                if (grid[r + 1][c] == color and
                    grid[r][c + 1] == color and
                    grid[r + 1][c + 1] == color):
                    blocks.append((r, c, color))
                    visited.update([(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)])
    return blocks

# Learn direction mapping from training data at module load time
def _learn_directions():
    try:
        with open('/tmp/rearc45/27d737e3.json') as f:
            task = json.load(f)
    except FileNotFoundError:
        return set(), set()

    up_right = set()
    down_left = set()

    for pair in task['train']:
        inp = pair['input']
        out = pair['output']
        H, W = len(inp), len(inp[0])
        bg = find_bg(inp)
        blocks = find_blocks(inp, bg)

        for r, c, color in blocks:
            # Check if this block emits up-right
            nr, nc = r - 1, c + 2
            if 0 <= nr < H and 0 <= nc < W and inp[nr][nc] == bg and out[nr][nc] == color:
                up_right.add(color)
            # Check if this block emits down-left
            nr, nc = r + 2, c - 1
            if 0 <= nr < H and 0 <= nc < W and inp[nr][nc] == bg and out[nr][nc] == color:
                down_left.add(color)

    return up_right, down_left

_UP_RIGHT_COLORS, _DOWN_LEFT_COLORS = _learn_directions()

def transform(grid):
    grid = [row[:] for row in grid]
    H, W = len(grid), len(grid[0])
    bg = find_bg(grid)
    blocks = find_blocks(grid, bg)

    for r, c, color in blocks:
        if color in _UP_RIGHT_COLORS:
            # Diagonal trail from top-right corner going up-right
            nr, nc = r - 1, c + 2
            while 0 <= nr < H and 0 <= nc < W:
                grid[nr][nc] = color
                nr -= 1
                nc += 1

        if color in _DOWN_LEFT_COLORS:
            # Diagonal trail from bottom-left corner going down-left
            nr, nc = r + 2, c - 1
            while 0 <= nr < H and 0 <= nc < W:
                grid[nr][nc] = color
                nr += 1
                nc -= 1

    return grid
