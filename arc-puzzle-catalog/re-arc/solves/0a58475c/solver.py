import copy
from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    out = copy.deepcopy(grid)
    
    color_count = Counter(v for row in grid for v in row)
    bg = color_count.most_common(1)[0][0]
    
    shape_colors = [c for c in color_count if c != bg]
    if not shape_colors:
        return out
    shape_color = shape_colors[0]
    
    visited = set()
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == shape_color and (r, c) not in visited:
                stack = [(r, c)]
                comp = set()
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited or cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if grid[cr][cc] != shape_color:
                        continue
                    visited.add((cr, cc))
                    comp.add((cr, cc))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            stack.append((cr + dr, cc + dc))
                components.append(comp)
    
    for comp in components:
        min_row = min(r for r, c in comp)
        if min_row == 0:
            continue
        
        tip_pixels = sorted([(r, c) for r, c in comp if r == min_row], key=lambda x: x[1])
        
        if len(tip_pixels) == 1:
            tip_r, tip_c = tip_pixels[0]
            
            left_in = tip_c > 0 and (tip_r + 1, tip_c - 1) in comp
            right_in = tip_c < cols - 1 and (tip_r + 1, tip_c + 1) in comp
            
            if left_in and not right_in:
                partner_c = tip_c - 1
            elif right_in and not left_in:
                partner_c = tip_c + 1
            else:
                partner_c = None
            
            if partner_c is not None:
                for r in range(tip_r - 1, -1, -1):
                    dist = tip_r - r
                    if dist % 2 == 1:
                        out[r][partner_c] = 4
                    else:
                        out[r][tip_c] = 4
            else:
                for r in range(tip_r - 1, -1, -1):
                    out[r][tip_c] = 4
        elif len(tip_pixels) >= 2:
            for _, tc in tip_pixels:
                for r in range(min_row - 1, -1, -1):
                    out[r][tc] = 4
    
    return out
