import json, sys
from collections import Counter

def solve(grid):
    H = len(grid)
    W = len(grid[0])
    
    color_counts = Counter()
    for row in grid:
        for c in row:
            color_counts[c] += 1
    bg = color_counts.most_common(1)[0][0]
    
    red_dots = []
    shape_pixels = []
    shape_color = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 2:
                red_dots.append((r, c))
            elif grid[r][c] != bg:
                shape_pixels.append((r, c))
                if shape_color is None:
                    shape_color = grid[r][c]
    
    v_axis = None; h_axis = None
    v_edge = None; h_edge = None
    
    for r, c in red_dots:
        if r == 0:
            v_axis = c; v_edge = 'top'
        elif r == H - 1:
            v_axis = c; v_edge = 'bottom'
        if c == 0:
            h_axis = r; h_edge = 'left'
        elif c == W - 1:
            h_axis = r; h_edge = 'right'
    
    out = [[bg] * W for _ in range(H)]
    
    axis_cells = set()
    quadrants = {}
    
    for r, c in shape_pixels:
        on_v = (v_axis is not None and c == v_axis)
        on_h = (h_axis is not None and r == h_axis)
        
        if on_v or on_h:
            axis_cells.add((r, c))
        else:
            v_side = 'left' if (v_axis is not None and c < v_axis) else ('right' if (v_axis is not None and c > v_axis) else 'all')
            h_side = 'top' if (h_axis is not None and r < h_axis) else ('bottom' if (h_axis is not None and r > h_axis) else 'all')
            key = (v_side, h_side)
            quadrants.setdefault(key, []).append((r, c))
    
    # Place axis cells
    for r, c in axis_cells:
        out[r][c] = shape_color
    
    # Draw vertical red line
    if v_axis is not None:
        shape_rows_on_v = sorted(r for r, c in axis_cells if c == v_axis)
        if shape_rows_on_v:
            if v_edge == 'top':
                for r in range(0, max(shape_rows_on_v) + 1):
                    if (r, v_axis) not in axis_cells:
                        out[r][v_axis] = 2
            else:
                for r in range(min(shape_rows_on_v), H):
                    if (r, v_axis) not in axis_cells:
                        out[r][v_axis] = 2
    
    # Draw horizontal red line
    if h_axis is not None:
        shape_cols_on_h = sorted(c for r, c in axis_cells if r == h_axis)
        if shape_cols_on_h:
            if h_edge == 'right':
                for c in range(min(shape_cols_on_h), W):
                    if (h_axis, c) not in axis_cells:
                        out[h_axis][c] = 2
            else:
                for c in range(0, max(shape_cols_on_h) + 1):
                    if (h_axis, c) not in axis_cells:
                        out[h_axis][c] = 2
    
    # Place quadrant cells
    reachable = set()
    if v_edge: reachable.add(v_edge)
    if h_edge: reachable.add(h_edge)
    
    for key, pixels in quadrants.items():
        v_side, h_side = key
        
        if h_side == 'all':
            row_target = 'min' if v_edge == 'top' else 'max'
        else:
            row_target = 'min' if h_side == 'top' else 'max'
        
        if v_side == 'all':
            col_target = 'min' if h_edge == 'left' else 'max'
        else:
            col_target = 'min' if v_side == 'left' else 'max'
        
        # Reachability check for two-axis case
        if v_axis is not None and h_axis is not None:
            h_corner = 'top' if row_target == 'min' else 'bottom'
            v_corner = 'left' if col_target == 'min' else 'right'
            if h_corner not in reachable and v_corner not in reachable:
                continue
        
        rows = [r for r, _ in pixels]
        cols = [c for _, c in pixels]
        
        row_shift = -min(rows) if row_target == 'min' else (H - 1) - max(rows)
        col_shift = -min(cols) if col_target == 'min' else (W - 1) - max(cols)
        
        for r, c in pixels:
            nr, nc = r + row_shift, c + col_shift
            if 0 <= nr < H and 0 <= nc < W:
                out[nr][nc] = shape_color
    
    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        status = 'PASS' if result == ex['output'] else 'FAIL'
        print(f"Train {i}: {status}")
        if status == 'FAIL':
            H = len(result)
            W = len(result[0])
            exp = ex['output']
            for r in range(H):
                for c in range(W):
                    if result[r][c] != exp[r][c]:
                        print(f"  Diff at ({r},{c}): got {result[r][c]}, expected {exp[r][c]}")
