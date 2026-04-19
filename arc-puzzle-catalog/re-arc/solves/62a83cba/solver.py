def transform(grid):
    import copy
    
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Find the background color (most common)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
    background = max(color_counts, key=color_counts.get)
    
    # Find bar regions (contiguous non-background on edges)
    # Bars contain the "bar color" (like 7 or 3)
    bar_positions = []  # List of (row, col, bar_color)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                # Check if this is on an edge or part of edge bar
                on_edge = (r == 0 or r == rows-1 or c == 0 or c == cols-1)
                if on_edge:
                    bar_positions.append((r, c, grid[r][c]))
    
    # Find marker pixels (non-background, non-bar colors in interior)
    bar_colors = set(bp[2] for bp in bar_positions)
    markers = []  # (row, col, marker_color)
    
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val != background and val not in bar_colors:
                markers.append((r, c, val))
    
    # For each marker, find the nearest bar and draw a line
    for mr, mc, marker_color in markers:
        # Find direction to bar
        # Check all 4 directions for bars
        directions = []
        
        # Up
        for r in range(mr-1, -1, -1):
            if grid[r][mc] in bar_colors:
                directions.append(('up', mr - r, r))
                break
        
        # Down
        for r in range(mr+1, rows):
            if grid[r][mc] in bar_colors:
                directions.append(('down', r - mr, r))
                break
        
        # Left
        for c in range(mc-1, -1, -1):
            if grid[mr][c] in bar_colors:
                directions.append(('left', mc - c, c))
                break
        
        # Right
        for c in range(mc+1, cols):
            if grid[mr][c] in bar_colors:
                directions.append(('right', c - mc, c))
                break
        
        if not directions:
            continue
            
        # Pick closest bar direction
        directions.sort(key=lambda x: x[1])
        best_dir = directions[0][0]
        
        # Draw line from bar to marker (exclusive of marker position)
        # Marker becomes color 6 (magenta)
        output[mr][mc] = 6
        
        if best_dir == 'up':
            bar_r = directions[0][2]
            for r in range(bar_r + 1, mr):
                output[r][mc] = marker_color
        elif best_dir == 'down':
            bar_r = directions[0][2]
            for r in range(mr + 1, bar_r):
                output[r][mc] = marker_color
        elif best_dir == 'left':
            bar_c = directions[0][2]
            for c in range(bar_c + 1, mc):
                output[mr][c] = marker_color
        elif best_dir == 'right':
            bar_c = directions[0][2]
            for c in range(mc + 1, bar_c):
                output[mr][c] = marker_color
    
    return output
