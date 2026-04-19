"""
ARC-AGI puzzle 774a13b4 solver

Pattern:
1. Find shapes (connected regions of non-background color)
2. For shapes WITH a marker inside:
   - Horizontal shapes (width >= height): expansion = min(left, right) - 1
   - Vertical shapes (height > width): expansion = width - 1
3. For shapes WITHOUT marker:
   - Horizontal: expand down, width = width - (height - 1), starting from left
4. Isolated markers expand to grid edge
"""

def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    output = [row[:] for row in input_grid]
    
    # Find background (most common color)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            v = input_grid[r][c]
            color_counts[v] = color_counts.get(v, 0) + 1
    bg = max(color_counts, key=color_counts.get)
    
    # Flood fill helper
    def flood_fill(sr, sc, color, visited):
        cells = []
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if (r, c) in visited:
                continue
            if input_grid[r][c] != color:
                continue
            visited.add((r, c))
            cells.append((r, c))
            stack.extend([(r+1,c), (r-1,c), (r,c+1), (r,c-1)])
        return cells
    
    # Find all shapes (size > 1)
    visited = set()
    shapes = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and input_grid[r][c] != bg:
                color = input_grid[r][c]
                cells = flood_fill(r, c, color, visited)
                if len(cells) > 1:
                    shapes.append((color, set(cells)))
    
    # Find isolated markers
    all_shape_cells = set()
    all_shape_bboxes = []
    for _, cells in shapes:
        all_shape_cells.update(cells)
        min_r = min(cell[0] for cell in cells)
        max_r = max(cell[0] for cell in cells)
        min_c = min(cell[1] for cell in cells)
        max_c = max(cell[1] for cell in cells)
        all_shape_bboxes.append((min_r, max_r, min_c, max_c))
    
    isolated_markers = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg and (r, c) not in all_shape_cells:
                inside_shape = False
                for (min_r, max_r, min_c, max_c) in all_shape_bboxes:
                    if min_r <= r <= max_r and min_c <= c <= max_c:
                        inside_shape = True
                        break
                if not inside_shape:
                    isolated_markers.append((r, c, input_grid[r][c]))
    
    # Process each shape
    for shape_color, cells in shapes:
        min_r = min(cell[0] for cell in cells)
        max_r = max(cell[0] for cell in cells)
        min_c = min(cell[1] for cell in cells)
        max_c = max(cell[1] for cell in cells)
        
        # Find marker (hole in shape)
        marker = None
        marker_color = None
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in cells:
                    marker = (r, c)
                    marker_color = input_grid[r][c]
                    break
            if marker:
                break
        
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        if marker:
            mr, mc = marker
            
            if width >= height:  # Horizontal shape -> vertical expansion
                left_count = sum(1 for (r, c) in cells if r == mr and c < mc)
                right_count = sum(1 for (r, c) in cells if r == mr and c > mc)
                
                expand_width = min(left_count, right_count) - 1
                if expand_width < 0:
                    expand_width = 0
                
                cells_above = sum(1 for (r, c) in cells if r < mr)
                cells_below = sum(1 for (r, c) in cells if r > mr)
                
                if cells_above >= cells_below:
                    for exp_r in range(max_r + 1, rows):
                        for dc in range(-expand_width, expand_width + 1):
                            c = mc + dc
                            if 0 <= c < cols:
                                output[exp_r][c] = marker_color if dc == 0 else shape_color
                else:
                    for exp_r in range(min_r - 1, -1, -1):
                        for dc in range(-expand_width, expand_width + 1):
                            c = mc + dc
                            if 0 <= c < cols:
                                output[exp_r][c] = marker_color if dc == 0 else shape_color
            
            else:  # Vertical shape -> horizontal expansion
                expand_height = width - 1  # Key fix: use width - 1
                
                cells_left = sum(1 for (r, c) in cells if c < mc)
                cells_right = sum(1 for (r, c) in cells if c > mc)
                
                if cells_left >= cells_right:
                    for exp_c in range(max_c + 1, cols):
                        for dr in range(-expand_height, expand_height + 1):
                            r = mr + dr
                            if 0 <= r < rows:
                                output[r][exp_c] = marker_color if dr == 0 else shape_color
                else:
                    for exp_c in range(min_c - 1, -1, -1):
                        for dr in range(-expand_height, expand_height + 1):
                            r = mr + dr
                            if 0 <= r < rows:
                                output[r][exp_c] = marker_color if dr == 0 else shape_color
        
        else:  # No marker - expand from edge
            if width >= height:  # Horizontal shape without marker
                # Expand down, width = width - (height - 1), from left edge
                expand_width = width - (height - 1)
                if expand_width > 0:
                    for exp_r in range(max_r + 1, rows):
                        for c in range(min_c, min_c + expand_width):
                            if 0 <= c < cols:
                                output[exp_r][c] = shape_color
    
    # Process isolated markers
    for mr, mc, marker_color in isolated_markers:
        dist_left = mc
        dist_right = cols - 1 - mc
        dist_top = mr
        dist_bottom = rows - 1 - mr
        
        if min(dist_left, dist_right) <= min(dist_top, dist_bottom):
            if dist_right <= dist_left:
                for c in range(mc + 1, cols):
                    output[mr][c] = marker_color
            else:
                for c in range(mc - 1, -1, -1):
                    output[mr][c] = marker_color
        else:
            if dist_bottom <= dist_top:
                for r in range(mr + 1, rows):
                    output[r][mc] = marker_color
            else:
                for r in range(mr - 1, -1, -1):
                    output[r][mc] = marker_color
    
    return output


if __name__ == "__main__":
    import json
    
    with open("/Users/evanpieser/re_arc_solves/774a13b4_task.json") as f:
        data = json.load(f)
    
    passed = 0
    for i, ex in enumerate(data["train"]):
        result = transform(ex["input"])
        expected = ex["output"]
        match = result == expected
        passed += match
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            diffs = []
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if expected[r][c] != result[r][c]:
                        diffs.append(f"({r},{c}): exp {expected[r][c]} got {result[r][c]}")
            print(f"  {len(diffs)} diffs:", diffs[:10], "..." if len(diffs) > 10 else "")
    
    print(f"\nPassed: {passed}/4")
