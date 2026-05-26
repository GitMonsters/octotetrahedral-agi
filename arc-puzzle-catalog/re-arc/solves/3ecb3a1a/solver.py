from collections import Counter

def transform(grid):
    """
    Solves ARC puzzle 3ecb3a1a.
    
    Rule: Diagonal dots approaching a strip bounce/reflect off it.
    The reflected portion uses a new color (0 typically).
    """
    grid = [row[:] for row in grid]  # Deep copy
    h, w = len(grid), len(grid[0])
    
    # Count colors to identify roles
    flat = [cell for row in grid for cell in row]
    color_counts = Counter(flat)
    colors_by_freq = [c for c, _ in color_counts.most_common()]
    
    if len(colors_by_freq) < 2:
        return grid
    
    background = colors_by_freq[0]
    strip_color = colors_by_freq[1]
    dot_color = colors_by_freq[2] if len(colors_by_freq) >= 3 else None
    
    if dot_color is None:
        return grid
    
    # Find the strip (either horizontal or vertical band)
    strip_rows = set()
    strip_cols = set()
    
    # Check for horizontal strip (all cells in a row are strip_color)
    for r in range(h):
        if all(grid[r][c] == strip_color for c in range(w)):
            strip_rows.add(r)
    
    # Check for vertical strip (all cells in a column are strip_color)
    for c in range(w):
        if all(grid[r][c] == strip_color for r in range(h)):
            strip_cols.add(c)
    
    # Find diagonal dots
    dots = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == dot_color:
                dots.append((r, c))
    
    if not dots or (not strip_rows and not strip_cols):
        return grid
    
    # Sort dots by row then column
    dots.sort()
    
    # Determine diagonal direction
    if len(dots) >= 2:
        dr = 1 if dots[-1][0] > dots[0][0] else -1
        dc = 1 if dots[-1][1] > dots[0][1] else -1
    else:
        return grid
    
    # Find reflection color (use 0, or find unused color)
    reflection_color = 0
    
    # Try extending from BOTH first and last dot to find which one approaches the strip
    # We'll try the first dot going backward (reverse direction)
    start_positions = [
        (dots[-1][0], dots[-1][1], dr, dc),  # From last dot, forward
        (dots[0][0], dots[0][1], -dr, -dc),  # From first dot, backward
    ]
    
    for start_r, start_c, dr_try, dc_try in start_positions:
        r, c = start_r, start_c
        dr_local = dr_try
        dc_local = dc_try
        
        if strip_cols:  # Vertical strip
            min_strip_col = min(strip_cols)
            max_strip_col = max(strip_cols)
            
            # Check if approaching strip
            if dc_local < 0 and c > max_strip_col:
                # Moving left toward strip
                bounced = False
                
                while 0 <= r < h and 0 <= c < w:
                    r += dr_local
                    c += dc_local
                    
                    if not (0 <= r < h and 0 <= c < w):
                        break
                    
                    # Check if we've reached the column adjacent to strip
                    if not bounced and c == max_strip_col + 1:
                        # Time to bounce - place this dot then flip
                        if grid[r][c] == background:
                            grid[r][c] = reflection_color
                        bounced = True
                        dc_local = -dc_local
                        continue
                    
                    if grid[r][c] == background:
                        grid[r][c] = reflection_color
                
                return grid  # Found and processed the bounce
            
            elif dc_local > 0 and c < min_strip_col:
                # Moving right toward strip
                bounced = False
                
                while 0 <= r < h and 0 <= c < w:
                    r += dr_local
                    c += dc_local
                    
                    if not (0 <= r < h and 0 <= c < w):
                        break
                    
                    # Check if we've reached the column adjacent to strip
                    if not bounced and c == min_strip_col - 1:
                        # Time to bounce
                        if grid[r][c] == background:
                            grid[r][c] = reflection_color
                        bounced = True
                        dc_local = -dc_local
                        continue
                    
                    if grid[r][c] == background:
                        grid[r][c] = reflection_color
                
                return grid  # Found and processed the bounce
        
        elif strip_rows:  # Horizontal strip
            min_strip_row = min(strip_rows)
            max_strip_row = max(strip_rows)
            
            # Check if approaching strip
            if dr_local < 0 and r > max_strip_row:
                # Moving up toward strip
                bounced = False
                
                while 0 <= r < h and 0 <= c < w:
                    r += dr_local
                    c += dc_local
                    
                    if not (0 <= r < h and 0 <= c < w):
                        break
                    
                    # Check if we've reached the row adjacent to strip
                    if not bounced and r == max_strip_row + 1:
                        # Time to bounce
                        if grid[r][c] == background:
                            grid[r][c] = reflection_color
                        bounced = True
                        dr_local = -dr_local
                        continue
                    
                    if grid[r][c] == background:
                        grid[r][c] = reflection_color
                
                return grid  # Found and processed the bounce
            
            elif dr_local > 0 and r < min_strip_row:
                # Moving down toward strip
                bounced = False
                
                while 0 <= r < h and 0 <= c < w:
                    r += dr_local
                    c += dc_local
                    
                    if not (0 <= r < h and 0 <= c < w):
                        break
                    
                    # Check if we've reached the row adjacent to strip
                    if not bounced and r == min_strip_row - 1:
                        # Time to bounce
                        if grid[r][c] == background:
                            grid[r][c] = reflection_color
                        bounced = True
                        dr_local = -dr_local
                        continue
                    
                    if grid[r][c] == background:
                        grid[r][c] = reflection_color
                
                return grid  # Found and processed the bounce
    
    return grid
