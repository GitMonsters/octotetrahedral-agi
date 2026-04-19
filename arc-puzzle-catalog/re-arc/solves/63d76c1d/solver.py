def transform(grid):
    import copy
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Find the background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg_color = Counter(flat).most_common(1)[0][0]
    
    # Find horizontal rails (connected segments of >=2 cells of same non-bg color)
    h_rails = []  # (row, start_col, end_col, color)
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r][c] != bg_color:
                start = c
                color = grid[r][c]
                count = 0
                while c < cols and grid[r][c] == color:
                    count += 1
                    c += 1
                if count >= 2:
                    h_rails.append((r, start, c-1, color))
            else:
                c += 1
    
    # Find vertical rails 
    v_rails = []  # (col, start_row, end_row, color)
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r][c] != bg_color:
                start = r
                color = grid[r][c]
                count = 0
                while r < rows and grid[r][c] == color:
                    count += 1
                    r += 1
                if count >= 2:
                    v_rails.append((c, start, r-1, color))
            else:
                r += 1
    
    # Cells covered by rails
    rail_cells = set()
    for r, start, end, color in h_rails:
        for c in range(start, end + 1):
            rail_cells.add((r, c))
    for c, start, end, color in v_rails:
        for r in range(start, end + 1):
            rail_cells.add((r, c))
    
    # Find markers: isolated non-background cells that are not part of rails
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg_color and (r, c) not in rail_cells:
                markers.append((r, c, grid[r][c]))
    
    marker_color = markers[0][2] if markers else 5
    
    # Process based on rail types
    if len(v_rails) >= 2:
        # Two vertical rails - process horizontally
        v_rails_sorted = sorted(v_rails, key=lambda x: x[0])  # Sort by column
        rail1 = v_rails_sorted[0]  # Left rail
        rail2 = v_rails_sorted[-1]  # Right rail
        
        c1, s1, e1, col1 = rail1  # col, start_row, end_row, color
        c2, s2, e2, col2 = rail2
        
        # Determine which rail the markers are associated with
        markers_in_rail2 = [m for m in markers if s2 <= m[0] <= e2]
        
        # For each marker in rail2's range, draw toward rail2 (right)
        for mr, mc, mcolor in markers_in_rail2:
            for c in range(mc + 1, c2):
                output[mr][c] = mcolor
            output[mr][mc] = 9
        
        # Get marker offsets from rail2 start
        marker_offsets = [m[0] - s2 for m in markers_in_rail2]
        
        # Fill rows in rail1 at the same offsets
        for offset in marker_offsets:
            fill_row = s1 + offset
            if s1 <= fill_row <= e1:
                for c in range(c1 + 1, cols):
                    if output[fill_row][c] == bg_color:
                        output[fill_row][c] = marker_color
    
    elif len(h_rails) >= 2:
        # Two horizontal rails - process vertically
        h_rails_sorted = sorted(h_rails, key=lambda x: x[0])  # Sort by row
        rail1 = h_rails_sorted[0]  # Top rail
        rail2 = h_rails_sorted[-1]  # Bottom rail
        
        r1, s1, e1, col1 = rail1  # row, start_col, end_col, color
        r2, s2, e2, col2 = rail2
        
        # For each marker, draw line from marker toward rail2
        for mr, mc, mcolor in markers:
            if s2 <= mc <= e2:
                for r in range(mr + 1, r2):
                    output[r][mc] = mcolor
                output[mr][mc] = 9
        
        # Fill vertical line connecting the two rails
        fill_col = s1  # Leftmost col of first rail
        for r in range(r1 + 1, r2 + 1):
            if output[r][fill_col] == bg_color:
                output[r][fill_col] = marker_color
    
    return output
