"""
ARC Puzzle 12df2b07 Solver

Pattern: Multiple objects of the same color with marker pixels indicating connection points.
Objects are translated to connect at their marker locations, then markers are removed.
"""

def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find background color (most common)
    unique, counts = np.unique(grid, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    # Find all non-background colors and their pixels
    colors = {}
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg_color:
                color = grid[r, c]
                if color not in colors:
                    colors[color] = []
                colors[color].append((r, c))
    
    if len(colors) == 0:
        return grid.tolist()
    
    # Find connected components for each color
    def flood_fill(start, color, visited):
        component = []
        stack = [start]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if grid[r, c] != color:
                continue
            visited.add((r, c))
            component.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                stack.append((r+dr, c+dc))
        return component
    
    components_by_color = {}
    for color, pixels in colors.items():
        visited = set()
        components = []
        for p in pixels:
            if p not in visited:
                comp = flood_fill(p, color, visited)
                if comp:
                    components.append(comp)
        components_by_color[color] = components
    
    # Find main object color (multiple large components) and marker color (small scattered pixels)
    main_color = None
    marker_color = None
    max_total_pixels = 0
    
    for color, comps in components_by_color.items():
        total_pixels = sum(len(c) for c in comps)
        if len(comps) >= 2 and total_pixels > max_total_pixels:
            # Check if this looks like main objects (bigger components)
            avg_size = total_pixels / len(comps)
            if avg_size >= 2:  # Main objects should have some size
                max_total_pixels = total_pixels
                main_color = color
    
    if main_color is None:
        # Take color with most total pixels
        for color, comps in components_by_color.items():
            total_pixels = sum(len(c) for c in comps)
            if total_pixels > max_total_pixels:
                max_total_pixels = total_pixels
                main_color = color
    
    # Find marker color - different from main and bg, usually small/sparse
    for color in colors:
        if color != main_color and color != bg_color:
            marker_color = color
            break
    
    if main_color is None:
        return grid.tolist()
    
    main_components = components_by_color[main_color]
    
    # For each component, find nearby marker pixels (within bounding box + margin)
    def get_bbox(comp):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        return min(rows), max(rows), min(cols), max(cols)
    
    def find_markers_near_component(comp, marker_pixels, margin=2):
        if not marker_pixels:
            return []
        r_min, r_max, c_min, c_max = get_bbox(comp)
        markers = []
        for mr, mc in marker_pixels:
            if r_min - margin <= mr <= r_max + margin and c_min - margin <= mc <= c_max + margin:
                markers.append((mr, mc))
        return markers
    
    marker_pixels = colors.get(marker_color, []) if marker_color else []
    
    # Associate markers with components
    comp_markers = []
    for comp in main_components:
        markers = find_markers_near_component(comp, marker_pixels)
        comp_markers.append((comp, markers))
    
    # Calculate translations to align markers
    # Strategy: Find pairs of components whose markers should overlap
    if len(main_components) >= 2 and marker_pixels:
        # Use first marker of each component as connection point
        # Translate all components relative to first one
        ref_comp, ref_markers = comp_markers[0]
        
        translations = [(0, 0)]  # First component stays put
        
        for i in range(1, len(comp_markers)):
            comp, markers = comp_markers[i]
            if markers and ref_markers:
                # Find closest marker pair
                best_dist = float('inf')
                best_tr = (0, 0)
                for rm in ref_markers:
                    for cm in markers:
                        # Translation to align cm to rm
                        dr = rm[0] - cm[0]
                        dc = rm[1] - cm[1]
                        dist = abs(dr) + abs(dc)
                        if dist < best_dist:
                            best_dist = dist
                            best_tr = (dr, dc)
                translations.append(best_tr)
            else:
                translations.append((0, 0))
    else:
        translations = [(0, 0)] * len(main_components)
    
    # Calculate output bounds
    all_translated = []
    for i, (comp, _) in enumerate(comp_markers):
        dr, dc = translations[i]
        for r, c in comp:
            all_translated.append((r + dr, c + dc))
    
    # Include non-translated markers that weren't matched
    if all_translated:
        min_r = min(p[0] for p in all_translated)
        max_r = max(p[0] for p in all_translated)
        min_c = min(p[1] for p in all_translated)
        max_c = max(p[1] for p in all_translated)
    else:
        min_r, max_r = 0, h - 1
        min_c, max_c = 0, w - 1
    
    # Create output grid
    out_h = max_r - min_r + 1
    out_w = max_c - min_c + 1
    
    # Ensure reasonable output size
    out_h = max(1, min(out_h, 30))
    out_w = max(1, min(out_w, 30))
    
    output = np.full((out_h, out_w), bg_color, dtype=int)
    
    # Draw translated components
    for i, (comp, _) in enumerate(comp_markers):
        dr, dc = translations[i]
        for r, c in comp:
            nr = r + dr - min_r
            nc = c + dc - min_c
            if 0 <= nr < out_h and 0 <= nc < out_w:
                output[nr, nc] = main_color
    
    return output.tolist()


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['12df2b07']
    
    print("Testing on all training examples:")
    all_pass = True
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        match = result == expected
        all_pass = all_pass and match
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0])}")
    
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
