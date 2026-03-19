"""
ARC Puzzle 11c849c6 Solver

Pattern: Find a large incomplete rectangle with holes. Find small marker patterns.
Fill the rectangle completely and place markers where the holes were.
"""

def transform(grid):
    import numpy as np
    grid = np.array(grid)
    rows, cols = grid.shape
    
    # Find background color (most common)
    unique, counts = np.unique(grid, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    # Find all connected components of non-background colors
    def find_rectangles(grid, bg):
        """Find bounding boxes of non-background regions"""
        non_bg = grid != bg
        if not np.any(non_bg):
            return []
        
        # Find bounding box of all non-bg pixels
        rows_with = np.any(non_bg, axis=1)
        cols_with = np.any(non_bg, axis=0)
        
        rectangles = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r,c] != bg and not visited[r,c]:
                    # BFS to find connected region
                    component = []
                    queue = [(r,c)]
                    visited[r,c] = True
                    while queue:
                        cr, cc = queue.pop(0)
                        component.append((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                                if not visited[nr,nc] and grid[nr,nc] != bg:
                                    visited[nr,nc] = True
                                    queue.append((nr,nc))
                    
                    if component:
                        min_r = min(p[0] for p in component)
                        max_r = max(p[0] for p in component)
                        min_c = min(p[1] for p in component)
                        max_c = max(p[1] for p in component)
                        rectangles.append({
                            'bounds': (min_r, min_c, max_r, max_c),
                            'pixels': component,
                            'area': (max_r - min_r + 1) * (max_c - min_c + 1),
                            'pixel_count': len(component)
                        })
        
        return rectangles
    
    rectangles = find_rectangles(grid, bg_color)
    
    if not rectangles:
        # Simple case: single marker on background
        non_bg_positions = np.argwhere(grid != bg_color)
        if len(non_bg_positions) == 1:
            r, c = non_bg_positions[0]
            marker_color = grid[r, c]
            # Create small output with marker repositioned
            h = rows - r
            w = cols - c
            out = np.full((h, min(w, 10)), bg_color)
            out[0, c] = marker_color
            return out.tolist()
        return grid.tolist()
    
    # Find the largest rectangle (by bounding box area) - this is the main shape
    # Also identify rectangles that have holes (incomplete)
    
    def analyze_rect(rect, grid, bg):
        min_r, min_c, max_r, max_c = rect['bounds']
        h, w = max_r - min_r + 1, max_c - min_c + 1
        subgrid = grid[min_r:max_r+1, min_c:max_c+1]
        
        # Find the dominant non-bg color in this region
        colors_in_region = subgrid[subgrid != bg]
        if len(colors_in_region) == 0:
            return None
        unique_c, counts_c = np.unique(colors_in_region, return_counts=True)
        main_color = unique_c[np.argmax(counts_c)]
        
        # Find holes (bg pixels inside the region)
        holes = []
        for i in range(h):
            for j in range(w):
                if subgrid[i,j] == bg:
                    holes.append((i, j))
        
        # Find marker colors (non-bg and non-main-color)
        marker_positions = {}
        for i in range(h):
            for j in range(w):
                if subgrid[i,j] != bg and subgrid[i,j] != main_color:
                    mc = subgrid[i,j]
                    if mc not in marker_positions:
                        marker_positions[mc] = []
                    marker_positions[mc].append((i, j))
        
        return {
            'bounds': rect['bounds'],
            'size': (h, w),
            'main_color': main_color,
            'holes': holes,
            'markers': marker_positions,
            'pixel_count': rect['pixel_count'],
            'area': rect['area']
        }
    
    analyzed = [analyze_rect(r, grid, bg_color) for r in rectangles]
    analyzed = [a for a in analyzed if a is not None]
    
    if not analyzed:
        return grid.tolist()
    
    # Find the main rectangle (largest with holes) and marker rectangles (smaller, may have special colors)
    # The main rectangle is the one we need to fill and mark
    
    # Sort by area
    analyzed.sort(key=lambda x: -x['area'])
    
    # The main shape typically has holes OR the largest one
    main_rect = None
    marker_rects = []
    
    for a in analyzed:
        if a['holes'] or (main_rect is None and a['area'] >= 4):
            if main_rect is None or a['area'] > main_rect['area']:
                if main_rect is not None:
                    marker_rects.append(main_rect)
                main_rect = a
            else:
                marker_rects.append(a)
        else:
            marker_rects.append(a)
    
    if main_rect is None:
        main_rect = analyzed[0]
        marker_rects = analyzed[1:]
    
    h, w = main_rect['size']
    main_color = main_rect['main_color']
    
    # Create output filled with main color
    output = np.full((h, w), main_color)
    
    # Find where to place markers
    # Markers from small rectangles indicate positions relative to holes
    
    # Collect all marker info from smaller rectangles
    all_markers = {}
    for mr in marker_rects:
        for color, positions in mr['markers'].items():
            if color not in all_markers:
                all_markers[color] = []
            for p in positions:
                all_markers[color].append(p)
    
    # Also check markers in main rectangle
    for color, positions in main_rect['markers'].items():
        if color not in all_markers:
            all_markers[color] = []
        for p in positions:
            all_markers[color].append(p)
    
    # Place markers at hole positions if we have marker info
    # Match holes to markers based on relative positions
    holes = main_rect['holes']
    
    if holes and all_markers:
        # Try to map marker positions to hole positions
        # Often the marker pattern mirrors the hole pattern
        for color, marker_pos in all_markers.items():
            for mp in marker_pos:
                # Place marker at corresponding position
                mr, mc = mp
                if 0 <= mr < h and 0 <= mc < w:
                    output[mr, mc] = color
    elif holes:
        # No explicit markers, but we may need to mark holes differently
        # Check if there's a single marker somewhere
        non_bg_non_main = []
        for r in range(rows):
            for c in range(cols):
                if grid[r,c] != bg_color and grid[r,c] != main_color:
                    non_bg_non_main.append((r, c, grid[r,c]))
        
        if non_bg_non_main and holes:
            # Map markers to holes based on relative offsets
            main_min_r, main_min_c, _, _ = main_rect['bounds']
            for r, c, color in non_bg_non_main:
                # Find which hole this marker might correspond to
                rel_r = r - main_min_r
                rel_c = c - main_min_c
                if 0 <= rel_r < h and 0 <= rel_c < w:
                    output[rel_r, rel_c] = color
    
    return output.tolist()
