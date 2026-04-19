def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    
    # Find the frame (look for a rectangular border of a different color)
    # First find all unique colors
    colors = set()
    for r in range(rows):
        for c in range(cols):
            colors.add(grid[r][c])
    
    # Find the bounding box of the frame
    # The frame color forms a rectangle border
    for frame_color in colors:
        positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == frame_color]
        if not positions:
            continue
        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        
        bg_color = None
        for color in colors:
            if color != frame_color:
                # Check if this could be interior
                for r in range(min_r+1, max_r):
                    for c in range(min_c+1, max_c):
                        if grid[r][c] == color:
                            bg_color = color
                            break
        
        if bg_color is None:
            continue
            
        # Find gaps in the frame (positions on border that should be frame_color but aren't)
        gaps = []
        for c in range(min_c, max_c+1):
            if grid[min_r][c] != frame_color:
                gaps.append((min_r, c, 'top'))
            if grid[max_r][c] != frame_color:
                gaps.append((max_r, c, 'bottom'))
        for r in range(min_r, max_r+1):
            if grid[r][min_c] != frame_color:
                gaps.append((r, min_c, 'left'))
            if grid[r][max_c] != frame_color:
                gaps.append((r, max_c, 'right'))
        
        if not gaps:
            continue
            
        # Fill interior with 9s
        for r in range(min_r+1, max_r):
            for c in range(min_c+1, max_c):
                if grid[r][c] == bg_color:
                    result[r][c] = 9
        
        # From gaps, draw lines outward
        for gr, gc, side in gaps:
            result[gr][gc] = 9
            
            if side == 'left':
                # Draw line going left
                for i in range(1, gc+1):
                    if 0 <= gc-i < cols:
                        result[gr][gc-i] = 9
            elif side == 'right':
                for i in range(1, cols-gc):
                    if 0 <= gc+i < cols:
                        result[gr][gc+i] = 9
            elif side == 'top':
                for i in range(1, gr+1):
                    if 0 <= gr-i < rows:
                        result[gr-i][gc] = 9
            elif side == 'bottom':
                for i in range(1, rows-gr):
                    if 0 <= gr+i < rows:
                        result[gr+i][gc] = 9
        
        # Draw diagonal lines from corners of gap regions
        gap_rows = sorted(set(g[0] for g in gaps if g[2] in ['left', 'right']))
        gap_cols = sorted(set(g[1] for g in gaps if g[2] in ['top', 'bottom']))
        
        for gr, gc, side in gaps:
            if side == 'left':
                # diagonal up-left and down-left from endpoints
                if gr == min(gap_rows) or len(gap_rows) == 1:
                    for i in range(1, min(gr+1, gc+1)+1):
                        if 0 <= gr-i < rows and 0 <= gc-i < cols:
                            result[gr-i][gc-i] = 9
                if gr == max(gap_rows) or len(gap_rows) == 1:
                    for i in range(1, min(rows-gr, gc+1)):
                        if 0 <= gr+i < rows and 0 <= gc-i < cols:
                            result[gr+i][gc-i] = 9
            elif side == 'right':
                if gr == min(gap_rows) or len(gap_rows) == 1:
                    for i in range(1, min(gr+1, cols-gc)+1):
                        if 0 <= gr-i < rows and 0 <= gc+i < cols:
                            result[gr-i][gc+i] = 9
                if gr == max(gap_rows) or len(gap_rows) == 1:
                    for i in range(1, min(rows-gr, cols-gc)):
                        if 0 <= gr+i < rows and 0 <= gc+i < cols:
                            result[gr+i][gc+i] = 9
            elif side == 'top':
                if gc == min(gap_cols) or len(gap_cols) == 1:
                    for i in range(1, min(gr+1, gc+1)+1):
                        if 0 <= gr-i < rows and 0 <= gc-i < cols:
                            result[gr-i][gc-i] = 9
                if gc == max(gap_cols) or len(gap_cols) == 1:
                    for i in range(1, min(gr+1, cols-gc)+1):
                        if 0 <= gr-i < rows and 0 <= gc+i < cols:
                            result[gr-i][gc+i] = 9
        
        break
    
    return result