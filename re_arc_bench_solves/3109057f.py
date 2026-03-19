def transform(grid):
    """
    ARC Puzzle 3109057f: Reflect a pattern in 4 directions around an X-marker.
    
    The X-marker is 5 pixels in an X shape (center + 4 diagonals).
    If no marker visible, center is derived from shape bounding box.
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find colored pixels
    colored_pixels = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                colored_pixels.append((r, c, grid[r][c]))
    
    if not colored_pixels:
        return result
    
    # Find X-marker pattern (5 pixels: center + 4 diagonal corners)
    marker_center = None
    marker_color = None
    
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            val = grid[r][c]
            if val != bg:
                if (grid[r-1][c-1] == val and grid[r-1][c+1] == val and
                    grid[r+1][c-1] == val and grid[r+1][c+1] == val):
                    marker_center = (r, c)
                    marker_color = val
                    break
        if marker_center:
            break
    
    if marker_center:
        cr, cc = marker_center
        # Exclude marker pixels from reflection
        colored_pixels = [(r, c, v) for r, c, v in colored_pixels if v != marker_color]
    else:
        # No visible marker - derive center from pattern bounding box
        max_r = max(p[0] for p in colored_pixels)
        max_c = max(p[1] for p in colored_pixels)
        cr = max_r + 2
        cc = max_c + 1
    
    # Reflect all colored pixels in 4 directions around center
    for r, c, color in colored_pixels:
        dr = r - cr
        dc = c - cc
        
        positions = [
            (cr + dr, cc + dc),   # original quadrant
            (cr + dr, cc - dc),   # horizontal flip
            (cr - dr, cc + dc),   # vertical flip
            (cr - dr, cc - dc),   # both flips
        ]
        
        for nr, nc in positions:
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr][nc] = color
    
    return result
