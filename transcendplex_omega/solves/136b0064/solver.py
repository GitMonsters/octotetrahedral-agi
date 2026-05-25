"""Solver for 136b0064 — shape-encoded path drawing"""
import json
from typing import List, Tuple

# Shape type mapping (normalized patterns → segment type)
# Type A: Red U → horizontal LEFT, length 2
# Type B: Purple V → vertical DOWN, length 2
# Type C: Blue → horizontal RIGHT, length 3
# Type D: Green → horizontal LEFT, length 4
SHAPE_MAP = {
    ((1,0,1),(1,0,1),(1,1,1)): ('H', 'LEFT', 2),   # Type A (7 cells)
    ((1,0,1),(0,1,0),(0,1,0)): ('V', 'DOWN', 2),    # Type B (4 cells)
    ((1,1,0),(1,0,1),(0,1,0)): ('H', 'RIGHT', 3),   # Type C (5 cells)
    ((1,1,1),(0,1,0),(1,0,1)): ('H', 'LEFT', 4),    # Type D (6 cells)
}

def solve(grid: List[List[int]]) -> List[List[int]]:
    H = len(grid)
    W = len(grid[0])
    
    # Find yellow separator column
    yellow_col = None
    for c in range(W):
        if all(grid[r][c] == 4 for r in range(H)):
            yellow_col = c
            break
    
    out_w = W - yellow_col - 1  # should be 7
    
    # Find white pixel position on right side (relative to output grid)
    white_r, white_c = 0, 0
    for r in range(H):
        for c in range(yellow_col + 1, W):
            if grid[r][c] == 5:
                white_r = r
                white_c = c - yellow_col - 1
    
    # Extract blocks (groups of non-empty rows separated by all-zero rows)
    blocks = []
    block_start = 0
    for r in range(H):
        if all(grid[r][c] == 0 for c in range(yellow_col)):
            if r > block_start:
                blocks.append((block_start, r - 1))
            block_start = r + 1
    if block_start < H:
        blocks.append((block_start, H - 1))
    
    # Extract shapes from each block (left: cols 0-2, right: cols 4-6)
    left_shapes = []
    right_shapes = []
    for rs, re in blocks:
        for side, shapes_list, col_start in [('L', left_shapes, 0), ('R', right_shapes, 4)]:
            shape = []
            for r in range(rs, re + 1):
                row = tuple(grid[r][col_start:col_start + 3])
                shape.append(row)
            color = max(v for row in shape for v in row)
            norm = tuple(tuple(1 if v > 0 else 0 for v in row) for row in shape)
            shapes_list.append((color, norm))
    
    # Build shape sequence: all left shapes, then all right shapes
    shape_seq = left_shapes + right_shapes
    
    # Initialize output grid
    result = [[0] * out_w for _ in range(H)]
    result[white_r][white_c] = 5  # Place white pixel
    
    # Execute path
    cur_r = white_r
    cur_c = white_c
    
    for color, norm in shape_seq:
        seg_type = SHAPE_MAP.get(norm)
        if seg_type is None:
            continue
        
        orient, direction, length = seg_type
        
        if orient == 'V':
            # Vertical: go down from current position
            for i in range(1, length + 1):
                nr = cur_r + i
                if 0 <= nr < H:
                    result[nr][cur_c] = color
            cur_r = cur_r + length
        else:
            # Horizontal: go in specified direction on next row
            next_r = cur_r + 1
            if direction == 'LEFT':
                for i in range(length):
                    nc = cur_c - i
                    if 0 <= nc < out_w and 0 <= next_r < H:
                        result[next_r][nc] = color
                cur_r = next_r
                cur_c = cur_c - length + 1
            else:  # RIGHT
                for i in range(length):
                    nc = cur_c + i
                    if 0 <= nc < out_w and 0 <= next_r < H:
                        result[next_r][nc] = color
                cur_r = next_r
                cur_c = cur_c + length - 1
    
    return result

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        exp = ex['output']
        match = result == exp
        print(f"Train {i}: {'PASS ✓' if match else 'FAIL'}")
        if not match:
            import numpy as np
            ra = np.array(result); ea = np.array(exp)
            diff = ra != ea
            for r in range(diff.shape[0]):
                for c in range(diff.shape[1]):
                    if diff[r][c]:
                        print(f"  ({r},{c}): exp={ea[r][c]}, got={ra[r][c]}")
