"""
ARC Puzzle 3d6c7249 Solver

Pattern: 
1. Grid is split into two halves (vertical or horizontal) with different backgrounds
2. One half contains TEMPLATE patterns (multi-colored shapes)
3. Other half contains MARKER pixels (single color indicating placement)
4. Templates are placed on output so their pixels of marker-color align with markers
5. Output uses the marker-half's background color
"""

def transform(grid):
    import numpy as np
    from collections import defaultdict
    
    grid = np.array(grid)
    h, w = grid.shape
    
    def get_dominant_color(region):
        unique, counts = np.unique(region, return_counts=True)
        return unique[np.argmax(counts)]
    
    def find_connected_components(half, bg):
        """Find all connected non-background pixel groups"""
        visited = set()
        components = []
        
        def bfs(start_r, start_c):
            component = []
            queue = [(start_r, start_c)]
            visited.add((start_r, start_c))
            while queue:
                r, c = queue.pop(0)
                component.append((r, c, half[r, c]))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < half.shape[0] and 0 <= nc < half.shape[1]:
                        if (nr, nc) not in visited and half[nr, nc] != bg:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            return component
        
        for r in range(half.shape[0]):
            for c in range(half.shape[1]):
                if (r, c) not in visited and half[r, c] != bg:
                    components.append(bfs(r, c))
        return components
    
    def get_bounding_box(component):
        rows = [p[0] for p in component]
        cols = [p[1] for p in component]
        return min(rows), min(cols), max(rows), max(cols)
    
    def extract_pattern(component, half):
        r0, c0, r1, c1 = get_bounding_box(component)
        pattern = np.full((r1-r0+1, c1-c0+1), -1)  # -1 = transparent
        for r, c, v in component:
            pattern[r-r0, c-c0] = v
        return pattern, r0, c0
    
    # Try both splits and find the one that works
    for split_type in ['vertical', 'horizontal']:
        if split_type == 'vertical':
            half1 = grid[:, :w//2].copy()
            half2 = grid[:, w//2:].copy()
        else:
            half1 = grid[:h//2, :].copy()
            half2 = grid[h//2:, :].copy()
        
        bg1 = get_dominant_color(half1)
        bg2 = get_dominant_color(half2)
        
        if bg1 == bg2:
            continue
            
        # Find components in each half
        comps1 = find_connected_components(half1, bg1)
        comps2 = find_connected_components(half2, bg2)
        
        if not comps1 or not comps2:
            continue
        
        # Determine which half has templates vs markers
        # Templates have multiple colors, markers have single color
        def count_colors(comps):
            colors = set()
            for comp in comps:
                for _, _, v in comp:
                    colors.add(v)
            return len(colors)
        
        colors1 = count_colors(comps1)
        colors2 = count_colors(comps2)
        
        if colors1 >= colors2:
            template_half, template_bg, template_comps = half1, bg1, comps1
            marker_half, marker_bg, marker_comps = half2, bg2, comps2
        else:
            template_half, template_bg, template_comps = half2, bg2, comps2
            marker_half, marker_bg, marker_comps = half1, bg1, comps1
        
        # Get marker color(s)
        marker_colors = set()
        for comp in marker_comps:
            for _, _, v in comp:
                marker_colors.add(v)
        
        # Build output on marker background
        out_h, out_w = marker_half.shape
        output = np.full((out_h, out_w), marker_bg)
        
        # For each template, find where its marker-colored pixels are
        # Then for each marker group, align template so colors match
        for t_comp in template_comps:
            pattern, t_r0, t_c0 = extract_pattern(t_comp, template_half)
            
            # Find marker-color pixels within this template
            marker_pixels_in_template = []
            for r in range(pattern.shape[0]):
                for c in range(pattern.shape[1]):
                    if pattern[r, c] in marker_colors:
                        marker_pixels_in_template.append((r, c, pattern[r, c]))
            
            if not marker_pixels_in_template:
                # No marker colors in template - just place at same relative position
                for r in range(pattern.shape[0]):
                    for c in range(pattern.shape[1]):
                        if pattern[r, c] != -1:
                            nr, nc = t_r0 + r, t_c0 + c
                            if 0 <= nr < out_h and 0 <= nc < out_w:
                                output[nr, nc] = pattern[r, c]
                continue
            
            # Find matching marker groups
            for m_comp in marker_comps:
                m_color = m_comp[0][2]  # marker color
                
                # Check if this marker group matches template's marker pixels
                # Try to align first marker pixel of template with first of marker group
                t_marker_pts = [(r, c) for r, c, v in marker_pixels_in_template if v == m_color]
                m_pts = [(r, c) for r, c, v in m_comp]
                
                if not t_marker_pts:
                    continue
                
                # Try alignment: offset so template markers align with actual markers
                # Use first points as anchor
                for t_r, t_c in t_marker_pts[:1]:
                    for m_r, m_c in m_pts[:1]:
                        offset_r = m_r - t_r
                        offset_c = m_c - t_c
                        
                        # Check if all template marker pixels align with marker group
                        match = True
                        for tr, tc in t_marker_pts:
                            if (tr + offset_r, tc + offset_c) not in m_pts:
                                match = False
                                break
                        
                        if match:
                            # Place template at this offset
                            for r in range(pattern.shape[0]):
                                for c in range(pattern.shape[1]):
                                    if pattern[r, c] != -1:
                                        nr = r + offset_r
                                        nc = c + offset_c
                                        if 0 <= nr < out_h and 0 <= nc < out_w:
                                            output[nr, nc] = pattern[r, c]
        
        return output.tolist()
    
    # Fallback
    return grid[:, :w//2].tolist()


if __name__ == "__main__":
    import json
    
    # Load task
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['3d6c7249']
    
    print("Testing on all training examples:\n")
    all_pass = True
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"Train {i}: {status}")
        
        if not match:
            all_pass = False
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0])}")
            # Show first difference
            for r in range(min(len(expected), len(result))):
                for c in range(min(len(expected[0]), len(result[0]))):
                    if expected[r][c] != result[r][c]:
                        print(f"  First diff at ({r},{c}): expected {expected[r][c]}, got {result[r][c]}")
                        break
                else:
                    continue
                break
    
    print(f"\n{'All tests passed!' if all_pass else 'Some tests failed.'}")
