def transform(grid):
    """
    For each rectangular object with a frame color and inner marker color:
    - Expand outward by the dimensions of the inner marker pattern
    - The frame color and inner color swap roles in the expanded region
    - New outer border uses frame color, expansion fill uses inner color
    """
    import copy
    
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Find the background color (most frequent)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg_color = Counter(flat).most_common(1)[0][0]
    
    # Find all rectangular objects
    visited = [[False]*cols for _ in range(rows)]
    
    def flood_fill(r, c):
        """Get all connected cells of non-background color"""
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc] or grid[cr][cc] == bg_color:
                continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells
    
    def get_bounding_box(cells):
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        return min_r, max_r, min_c, max_c
    
    # Find all objects
    objects = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg_color:
                cells = flood_fill(r, c)
                if cells:
                    objects.append(cells)
    
    # Process each object
    for obj_cells in objects:
        min_r, max_r, min_c, max_c = get_bounding_box(obj_cells)
        obj_set = set(obj_cells)
        
        # Get colors in this object
        colors = {}
        for r, c in obj_cells:
            color = grid[r][c]
            if color not in colors:
                colors[color] = []
            colors[color].append((r, c))
        
        if len(colors) < 2:
            # Single color object - expand with same color
            frame_color = list(colors.keys())[0]
            inner_color = frame_color
            inner_h = inner_w = 1
        else:
            # Find frame color (on the boundary of bounding box) and inner color
            frame_color = None
            inner_color = None
            
            for color, cells in colors.items():
                on_boundary = any(r == min_r or r == max_r or c == min_c or c == max_c 
                                  for r, c in cells)
                if on_boundary:
                    frame_color = color
                else:
                    inner_color = color
            
            # Fallback: if all colors on boundary, use less common as inner
            if inner_color is None:
                sorted_colors = sorted(colors.items(), key=lambda x: len(x[1]))
                inner_color = sorted_colors[0][0]
                frame_color = sorted_colors[-1][0]
        
        # Get dimensions of inner marker pattern
        inner_cells = colors.get(inner_color, [])
        if inner_cells:
            inner_min_r = min(r for r,c in inner_cells)
            inner_max_r = max(r for r,c in inner_cells)
            inner_min_c = min(c for r,c in inner_cells)
            inner_max_c = max(c for r,c in inner_cells)
            inner_h = inner_max_r - inner_min_r + 1
            inner_w = inner_max_c - inner_min_c + 1
        else:
            inner_h = inner_w = 1
        
        # Expand by inner_h in vertical and inner_w in horizontal
        new_min_r = max(0, min_r - inner_h)
        new_max_r = min(rows - 1, max_r + inner_h)
        new_min_c = max(0, min_c - inner_w)
        new_max_c = min(cols - 1, max_c + inner_w)
        
        # Fill the expanded region
        for r in range(new_min_r, new_max_r + 1):
            for c in range(new_min_c, new_max_c + 1):
                in_original = (min_r <= r <= max_r and min_c <= c <= max_c)
                
                # Distance from original boundary
                dist_top = min_r - r if r < min_r else 0
                dist_bot = r - max_r if r > max_r else 0
                dist_left = min_c - c if c < min_c else 0
                dist_right = c - max_c if c > max_c else 0
                
                if not in_original:
                    # In expanded region
                    # Outer ring (distance == inner_h or inner_w) gets frame color
                    on_outer = (dist_top == inner_h or dist_bot == inner_h or 
                                dist_left == inner_w or dist_right == inner_w)
                    if on_outer:
                        output[r][c] = frame_color
                    else:
                        output[r][c] = inner_color
        
        # Swap colors in original region: frame -> inner, inner -> frame
        for r, c in obj_cells:
            if grid[r][c] == frame_color:
                output[r][c] = inner_color
            elif grid[r][c] == inner_color:
                output[r][c] = frame_color
    
    return output


if __name__ == "__main__":
    import json
    
    # Load test data
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    puzzle = data['72a46440']
    
    all_pass = True
    for i, ex in enumerate(puzzle['train']):
        result = transform(ex['input'])
        expected = ex['output']
        
        match = result == expected
        print(f"Train {i+1}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            all_pass = False
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
    
    print(f"\nAll tests passed: {all_pass}")
