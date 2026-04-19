from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    flat = sum(grid, [])
    bg = Counter(flat).most_common(1)[0][0]
    
    pixels = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not pixels:
        return [row[:] for row in grid]
    
    r_off = pixels[0][0] % 2
    c_off = pixels[0][1] % 2
    
    idx_pixels = [((r - r_off) // 2, (c - c_off) // 2, v) for r, c, v in pixels]
    
    full_R = (rows - r_off + 1) // 2
    full_C = (cols - c_off + 1) // 2
    
    r_coords = [r for r, c, v in idx_pixels]
    c_coords = [c for r, c, v in idx_pixels]
    r_center = sum(r_coords) / len(r_coords)
    c_center = sum(c_coords) / len(c_coords)
    
    at_top = r_center < full_R / 2
    at_left = c_center < full_C / 2
    
    if at_top and at_left:
        corner_r, corner_c = min(r_coords), min(c_coords)
    elif at_top and not at_left:
        corner_r, corner_c = min(r_coords), max(c_coords)
    elif not at_top and at_left:
        corner_r, corner_c = max(r_coords), min(c_coords)
    else:
        corner_r, corner_c = max(r_coords), max(c_coords)
    
    at_full_corner = (corner_r in (0, full_R - 1)) and (corner_c in (0, full_C - 1))
    
    if at_full_corner:
        grid_r_start, grid_c_start = 0, 0
        grid_R, grid_C = full_R, full_C
    else:
        max_shell = 0
        for r, c, v in idx_pixels:
            s = min(abs(r - corner_r), abs(c - corner_c))
            max_shell = max(max_shell, s)
        
        grid_size = 2 * (max_shell + 1)
        grid_R = grid_C = grid_size
        
        if at_top:
            grid_r_start = corner_r
        else:
            grid_r_start = corner_r - grid_R + 1
        if at_left:
            grid_c_start = corner_c
        else:
            grid_c_start = corner_c - grid_C + 1
    
    shell_corners = {}
    shell_edges = {}
    for r, c, v in idx_pixels:
        local_r = r - grid_r_start
        local_c = c - grid_c_start
        dr = min(local_r, grid_R - 1 - local_r)
        dc = min(local_c, grid_C - 1 - local_c)
        shell = min(dr, dc)
        if dr == dc:
            shell_corners[shell] = v
        else:
            shell_edges[shell] = v
    
    output = [[bg] * cols for _ in range(rows)]
    
    for local_r in range(grid_R):
        for local_c in range(grid_C):
            dr = min(local_r, grid_R - 1 - local_r)
            dc = min(local_c, grid_C - 1 - local_c)
            shell = min(dr, dc)
            is_corner = (dr == dc)
            val = shell_corners.get(shell, bg) if is_corner else shell_edges.get(shell, bg)
            if val != bg:
                orig_r = r_off + (grid_r_start + local_r) * 2
                orig_c = c_off + (grid_c_start + local_c) * 2
                if 0 <= orig_r < rows and 0 <= orig_c < cols:
                    output[orig_r][orig_c] = val
    
    return output
