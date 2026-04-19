from collections import Counter

def transform(grid):
    grid = [row[:] for row in grid]  # Copy
    h, w = len(grid), len(grid[0])
    
    # Find background color (most common)
    all_cells = [c for row in grid for c in row]
    bg = Counter(all_cells).most_common(1)[0][0]
    
    # Find the filled rectangular box (largest solid non-bg region)
    box_color = None
    box_top, box_bottom, box_left, box_right = None, None, None, None
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                color = grid[r][c]
                r2, c2 = r, c
                while c2 + 1 < w and grid[r][c2 + 1] == color:
                    c2 += 1
                while r2 + 1 < h:
                    if all(grid[r2 + 1][cc] == color for cc in range(c, c2 + 1)):
                        r2 += 1
                    else:
                        break
                area = (r2 - r + 1) * (c2 - c + 1)
                if box_color is None or area > (box_bottom - box_top + 1) * (box_right - box_left + 1):
                    is_solid = all(grid[rr][cc] == color for rr in range(r, r2 + 1) for cc in range(c, c2 + 1))
                    if is_solid and area > 4:
                        box_color = color
                        box_top, box_bottom = r, r2
                        box_left, box_right = c, c2
    
    if box_color is None:
        return grid
    
    # Project edge markers into the box
    for r in range(box_top, box_bottom + 1):
        if grid[r][0] != bg:
            grid[r][box_left] = grid[r][0]
        if grid[r][w-1] != bg:
            grid[r][box_right] = grid[r][w-1]
    
    for c in range(box_left, box_right + 1):
        if grid[0][c] != bg:
            grid[box_top][c] = grid[0][c]
        if grid[h-1][c] != bg:
            grid[box_bottom][c] = grid[h-1][c]
    
    return grid
