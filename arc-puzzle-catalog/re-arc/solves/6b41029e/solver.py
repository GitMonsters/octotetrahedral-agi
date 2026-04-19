from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    flat = sum(grid, [])
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find non-bg pixels
    pixels = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                pixels.append((r, c, grid[r][c]))
    
    if not pixels:
        return [row[:] for row in grid]
    
    r_min = min(p[0] for p in pixels)
    r_max = max(p[0] for p in pixels)
    c_min = min(p[1] for p in pixels)
    c_max = max(p[1] for p in pixels)
    
    height = r_max - r_min + 1
    width = c_max - c_min + 1
    side = max(height, width)
    
    # Expand to square, centering and adjusting for bounds
    r_extra = side - height
    r_before = r_extra // 2
    r_after = r_extra - r_before
    new_r_min = r_min - r_before
    new_r_max = r_max + r_after
    if new_r_min < 0:
        new_r_max -= new_r_min
        new_r_min = 0
    elif new_r_max >= rows:
        new_r_min -= (new_r_max - rows + 1)
        new_r_max = rows - 1
    
    c_extra = side - width
    c_before = c_extra // 2
    c_after = c_extra - c_before
    new_c_min = c_min - c_before
    new_c_max = c_max + c_after
    if new_c_min < 0:
        new_c_max -= new_c_min
        new_c_min = 0
    elif new_c_max >= cols:
        new_c_min -= (new_c_max - cols + 1)
        new_c_max = cols - 1
    
    cr = (new_r_min + new_r_max) / 2.0
    cc = (new_c_min + new_c_max) / 2.0
    
    # Apply 4-fold rotation
    output = [[bg] * cols for _ in range(rows)]
    
    for r, c, v in pixels:
        for angle in range(4):
            if angle == 0:
                nr, nc = float(r), float(c)
            elif angle == 1:
                nr = cr + (c - cc)
                nc = cc - (r - cr)
            elif angle == 2:
                nr = 2 * cr - r
                nc = 2 * cc - c
            else:
                nr = cr - (c - cc)
                nc = cc + (r - cr)
            
            nr, nc = int(round(nr)), int(round(nc))
            if 0 <= nr < rows and 0 <= nc < cols:
                output[nr][nc] = v
    
    return output
