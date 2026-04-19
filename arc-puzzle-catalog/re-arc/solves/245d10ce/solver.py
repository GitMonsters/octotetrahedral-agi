def transform(grid):
    """
    Puzzle 245d10ce: Find rectangular frame containing 8s+fill pattern.
    For scattered 8-clusters outside, convert isolated 8s to fill color
    based on the template pattern (8s that aren't part of closed rectangular frames
    become fill color if they're in "interior" positions).
    """
    import copy
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all colors present
    colors = set(flat)
    colors.discard(bg)
    colors.discard(8)  # 8 is the marker color
    
    # Find the fill color (non-bg, non-8, typically 6, 5, 1, or 2)
    fill_color = None
    frame_color = None
    
    # Find rectangular frame made of some color containing 8s inside
    def find_frame():
        # Look for rectangular boundaries
        for color in colors:
            positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
            if len(positions) < 8:
                continue
            # Check if forms a rectangle boundary
            min_r = min(p[0] for p in positions)
            max_r = max(p[0] for p in positions)
            min_c = min(p[1] for p in positions)
            max_c = max(p[1] for p in positions)
            
            # Check if it's a closed rectangle with 8s inside
            is_frame = True
            has_8_inside = False
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if r == min_r or r == max_r or c == min_c or c == max_c:
                        if grid[r][c] != color and grid[r][c] != 8:
                            is_frame = False
                    else:
                        if grid[r][c] == 8:
                            has_8_inside = True
            
            if is_frame and has_8_inside:
                return color, (min_r, max_r, min_c, max_c)
        return None, None
    
    frame_color, frame_bounds = find_frame()
    
    if frame_color is None:
        # Alternative: find frame made of 8s containing fill color inside
        positions_8 = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
        
        # Find connected rectangular frames of 8s
        def get_8_rectangles():
            # Find groups of 8s that form rectangular boundaries
            visited = set()
            rectangles = []
            
            for start_r, start_c in positions_8:
                if (start_r, start_c) in visited:
                    continue
                # BFS to find connected component
                component = []
                queue = [(start_r, start_c)]
                visited.add((start_r, start_c))
                while queue:
                    r, c = queue.pop(0)
                    component.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == 8:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                
                if len(component) >= 8:
                    min_r = min(p[0] for p in component)
                    max_r = max(p[0] for p in component)
                    min_c = min(p[1] for p in component)
                    max_c = max(p[1] for p in component)
                    # Check if forms closed rectangle
                    is_rect = True
                    for c in range(min_c, max_c + 1):
                        if (min_r, c) not in component and grid[min_r][c] != 8:
                            is_rect = False
                        if (max_r, c) not in component and grid[max_r][c] != 8:
                            is_rect = False
                    for r in range(min_r, max_r + 1):
                        if (r, min_c) not in component and grid[r][min_c] != 8:
                            is_rect = False
                        if (r, max_c) not in component and grid[r][max_c] != 8:
                            is_rect = False
                    
                    if is_rect and max_r - min_r >= 2 and max_c - min_c >= 2:
                        rectangles.append((component, (min_r, max_r, min_c, max_c)))
            
            return rectangles
        
        rects = get_8_rectangles()
        if rects:
            # Find the one with fill color inside
            for comp, bounds in rects:
                min_r, max_r, min_c, max_c = bounds
                for r in range(min_r + 1, max_r):
                    for c in range(min_c + 1, max_c):
                        if grid[r][c] != 8 and grid[r][c] != bg:
                            fill_color = grid[r][c]
                            frame_bounds = bounds
                            frame_color = 8
                            break
                    if fill_color:
                        break
                if fill_color:
                    break
    
    if frame_bounds is None:
        return grid
    
    min_r, max_r, min_c, max_c = frame_bounds
    
    # Find fill color from inside frame if not found
    if fill_color is None:
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if grid[r][c] != 8 and grid[r][c] != bg and grid[r][c] != frame_color:
                    fill_color = grid[r][c]
                    break
            if fill_color:
                break
    
    if fill_color is None:
        return grid
    
    # Extract template pattern: relative positions of fill_color to 8s inside frame
    # Get the interior pattern
    template_fill = []  # (relative_row, relative_col) of fill color
    template_8s = []    # positions of 8s that aren't on frame boundary
    
    frame_interior_8s = set()
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == 8:
                # Check if it's interior (not on perimeter of frame)
                if r > min_r and r < max_r and c > min_c and c < max_c:
                    frame_interior_8s.add((r - min_r, c - min_c))
            if grid[r][c] == fill_color:
                template_fill.append((r - min_r, c - min_c))
    
    # For scattered 8 clusters outside the frame, fill in based on template
    # Find all 8s not in the frame region
    positions_8_outside = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                if not (min_r <= r <= max_r and min_c <= c <= max_c):
                    positions_8_outside.append((r, c))
    
    # For each outside 8 cluster, determine fill pattern
    # The template shows: certain cells relative to 8-patterns should be fill_color
    
    # Compute the offset pattern from template
    # For each 8 inside frame boundary but not on perimeter, 
    # find relative fill positions
    
    # Simple approach: for each scattered 8, check neighbors and apply similar fill pattern
    
    # Get interior fill pattern relative to frame
    frame_h = max_r - min_r + 1
    frame_w = max_c - min_c + 1
    
    # Create output grid
    output = copy.deepcopy(grid)
    
    # Find L-shaped or corner 8 patterns outside and apply template fill
    # Get connected components of 8s outside frame
    visited = set()
    
    def get_component(start):
        comp = []
        queue = [start]
        visited.add(start)
        while queue:
            r, c = queue.pop(0)
            comp.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in visited and grid[nr][nc] == 8:
                        if not (min_r <= nr <= max_r and min_c <= nc <= max_c):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
        return comp
    
    components = []
    for r, c in positions_8_outside:
        if (r, c) not in visited:
            comp = get_component((r, c))
            components.append(comp)
    
    # For the template, find the fill positions relative to 8 positions
    template_8_to_fill = []
    for fr, fc in template_fill:
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if (fr + dr, fc + dc) in [(r - min_r, c - min_c) for r in range(min_r, max_r+1) 
                                           for c in range(min_c, max_c+1) if grid[r][c] == 8]:
                    template_8_to_fill.append((dr, dc, fr, fc))
    
    # Alternative simpler approach: look at the shape of 8 clusters
    # and fill in cells that would make them match the template's 8+fill pattern
    
    # Check what the template looks like - 8s on perimeter, fill inside
    # For L-shaped outside clusters, add fill color to make similar pattern
    
    # For each small component, add fill colors at positions that
    # correspond to where fill_color appears relative to 8s in template
    
    # Find relative pattern: in template, fill appears adjacent to 8s
    # at specific relative offsets
    fill_offsets = set()
    for fr, fc in template_fill:
        actual_r, actual_c = min_r + fr, min_c + fc
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = actual_r + dr, actual_c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 8:
                fill_offsets.add((-dr, -dc))  # offset from 8 to fill
    
    # Apply fill to each component
    for comp in components:
        if len(comp) <= 4:  # small clusters
            # For each 8 in component, check if fill should be added nearby
            comp_set = set(comp)
            for r, c in comp:
                for dr, dc in fill_offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if output[nr][nc] == bg:
                            # Check if this position should get fill
                            # Based on template pattern
                            output[nr][nc] = fill_color
                        elif output[nr][nc] == 8 and (nr, nc) in comp_set:
                            # Some 8s in cluster become fill color
                            # Check if this 8 is "interior" to the L-shape
                            pass
    
    # Final approach: replicate fill pattern from template to matching 8-shapes
    # The fill pattern relative to the frame's bounding box
    
    # Get normalized template: positions of 8s and fill relative to top-left
    template_pattern = {}  # (rel_r, rel_c) -> value (8 or fill_color)
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] in [8, fill_color]:
                template_pattern[(r - min_r, c - min_c)] = grid[r][c]
    
    # For each component, find matching template orientation and apply fill
    # This is complex - let's try the direct "shadow" approach
    
    # Reset output
    output = [list(row) for row in grid]
    
    # Find the direction of fill relative to 8 boundary
    # In template, fill is at bottom-right or specific corner
    
    # Count fill positions relative to 8 neighbors
    fill_directions = Counter()
    for fr, fc in template_fill:
        actual_r, actual_c = min_r + fr, min_c + fc
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = actual_r + dr, actual_c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 8:
                fill_directions[(dr, dc)] += 1
    
    # The fill is in direction opposite to 8s
    # For each scattered 8, add fill in that direction
    
    # For each scattered 8 not in frame, add fill nearby
    for r, c in positions_8_outside:
        for (d_from_8_r, d_from_8_c), count in fill_directions.items():
            # fill is at (r - d_from_8_r, c - d_from_8_c) relative to 8 at (r,c)
            nr, nc = r - d_from_8_r, c - d_from_8_c
            if 0 <= nr < rows and 0 <= nc < cols:
                if output[nr][nc] == bg:
                    output[nr][nc] = fill_color
    
    return output
