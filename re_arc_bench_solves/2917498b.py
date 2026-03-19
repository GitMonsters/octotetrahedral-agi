"""
Puzzle 2917498b: Find a frame and a pattern, tile the pattern into the frame.
- Frame: rectangular boundary defined by two parallel lines with corner markers
- Pattern: a smaller object with non-background colors
- Output: the frame filled with the pattern tiled/repeated
"""

def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find background color (most common)
    colors, counts = np.unique(grid, return_counts=True)
    bg = colors[np.argmax(counts)]
    
    # Find all non-background objects by connected components
    def find_objects(g, bg_color):
        visited = set()
        objects = []
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if g[i,j] != bg_color and (i,j) not in visited:
                    # BFS to find connected component
                    obj_cells = []
                    queue = [(i,j)]
                    visited.add((i,j))
                    while queue:
                        r, c = queue.pop(0)
                        obj_cells.append((r, c, g[r,c]))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                                if (nr,nc) not in visited and g[nr,nc] != bg_color:
                                    visited.add((nr,nc))
                                    queue.append((nr,nc))
                    objects.append(obj_cells)
        return objects
    
    objects = find_objects(grid, bg)
    
    # Separate frame from pattern - frame is larger, has more distinct structure
    # Frame typically has corner markers (same color at opposite ends of lines)
    
    def get_bbox(obj):
        rows = [c[0] for c in obj]
        cols = [c[1] for c in obj]
        return min(rows), max(rows), min(cols), max(cols)
    
    def obj_to_grid(obj):
        r1, r2, c1, c2 = get_bbox(obj)
        h, w = r2-r1+1, c2-c1+1
        g = np.full((h, w), bg)
        for r, c, v in obj:
            g[r-r1, c-c1] = v
        return g
    
    # Find frame: look for rectangular structure with corner markers
    # Frame has two parallel edges with markers at corners
    frame_obj = None
    pattern_obj = None
    
    # Heuristic: frame is typically more elongated (line-like) or has larger bbox
    # Pattern is more compact
    
    for obj in objects:
        bbox = get_bbox(obj)
        r1, r2, c1, c2 = bbox
        height, width = r2-r1+1, c2-c1+1
        cells = len(obj)
        area = height * width
        density = cells / area
        
        # Frame typically has low density (sparse) and larger dimensions
        if frame_obj is None or (area > get_bbox(frame_obj)[1]-get_bbox(frame_obj)[0]+1) * (get_bbox(frame_obj)[3]-get_bbox(frame_obj)[2]+1):
            if density < 0.5 and (height > 3 or width > 3):
                # Likely a frame
                if frame_obj is not None:
                    pattern_obj = frame_obj
                frame_obj = obj
            else:
                pattern_obj = obj
    
    # If we have multiple objects, separate frame from pattern more carefully
    if len(objects) >= 2:
        # Sort by area
        sorted_objs = sorted(objects, key=lambda o: (get_bbox(o)[1]-get_bbox(o)[0]+1) * (get_bbox(o)[3]-get_bbox(o)[2]+1), reverse=True)
        
        # Frame candidates: have linear structure or corner markers
        frame_obj = None
        pattern_objs = []
        
        for obj in sorted_objs:
            bbox = get_bbox(obj)
            r1, r2, c1, c2 = bbox
            cells = len(obj)
            height, width = r2-r1+1, c2-c1+1
            
            # Check if it's a frame (has corner markers - same color at corners)
            colors_at_obj = set(c[2] for c in obj)
            
            # Check for frame structure: sparse with lines
            is_sparse = cells < height * width * 0.5
            
            if frame_obj is None and is_sparse and (height > 3 or width > 3):
                frame_obj = obj
            else:
                pattern_objs.append(obj)
        
        if pattern_objs:
            pattern_obj = pattern_objs[0]
    
    if frame_obj is None or pattern_obj is None:
        return grid.tolist()
    
    # Get frame bounds and identify edges
    frame_bbox = get_bbox(frame_obj)
    fr1, fr2, fc1, fc2 = frame_bbox
    frame_h, frame_w = fr2-fr1+1, fc2-fc1+1
    
    # Create frame grid
    frame_grid = np.full((frame_h, frame_w), bg)
    for r, c, v in frame_obj:
        frame_grid[r-fr1, c-fc1] = v
    
    # Get pattern grid
    pattern_grid = obj_to_grid(pattern_obj)
    ph, pw = pattern_grid.shape
    
    # Create output by tiling pattern into frame dimensions
    output = np.full((frame_h, frame_w), bg)
    
    # Tile the pattern
    for i in range(frame_h):
        for j in range(frame_w):
            pi = i % ph
            pj = j % pw
            if pattern_grid[pi, pj] != bg:
                output[i, j] = pattern_grid[pi, pj]
    
    # Overlay frame on top (frame has priority)
    for r, c, v in frame_obj:
        output[r-fr1, c-fc1] = v
    
    return output.tolist()


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['2917498b']
    
    print("Testing on training examples:")
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        match = result == expected
        print(f"Example {i+1}: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"  Expected: {len(expected)}x{len(expected[0])}")
            print(f"  Got: {len(result)}x{len(result[0])}")
            print("  Expected output:")
            for row in expected:
                print(f"    {row}")
            print("  Got:")
            for row in result:
                print(f"    {row}")
