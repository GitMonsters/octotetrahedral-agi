"""
ARC Puzzle 5b834496 Solver

Pattern: There's a template shape with multiple colors, and incomplete copies
scattered around. Complete each incomplete copy by filling in the missing colors
based on the template's relative positions.
"""

def transform(grid: list[list[int]]) -> list[list[int]]:
    import copy
    
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)
    
    # Find background color (most common)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
    background = max(color_counts, key=color_counts.get)
    
    # Find all non-background colors and their positions
    non_bg_positions = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                color = grid[r][c]
                if color not in non_bg_positions:
                    non_bg_positions[color] = []
                non_bg_positions[color].append((r, c))
    
    if len(non_bg_positions) < 2:
        return result
    
    # Find the most common non-background color (the shape/frame color)
    frame_color = max(non_bg_positions.keys(), key=lambda c: len(non_bg_positions[c]))
    marker_colors = [c for c in non_bg_positions if c != frame_color]
    
    # Find connected components of the frame color
    def get_component(start_r, start_c, visited):
        component = set()
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] != frame_color:
                continue
            visited.add((r, c))
            component.add((r, c))
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if (dr, dc) != (0, 0):
                        stack.append((r + dr, c + dc))
        return component
    
    # Find all frame components
    visited = set()
    frame_components = []
    for r, c in non_bg_positions.get(frame_color, []):
        if (r, c) not in visited:
            comp = get_component(r, c, visited)
            if comp:
                frame_components.append(comp)
    
    # For each component, find which marker colors are nearby (within bounding box + margin)
    def get_bbox(positions):
        rs = [p[0] for p in positions]
        cs = [p[1] for p in positions]
        return min(rs), max(rs), min(cs), max(cs)
    
    def get_nearby_markers(component, margin=3):
        min_r, max_r, min_c, max_c = get_bbox(component)
        markers = {}
        for color in marker_colors:
            for r, c in non_bg_positions[color]:
                if min_r - margin <= r <= max_r + margin and min_c - margin <= c <= max_c + margin:
                    if color not in markers:
                        markers[color] = []
                    markers[color].append((r, c))
        return markers
    
    # Find the template (component with most marker colors nearby)
    component_markers = []
    for comp in frame_components:
        markers = get_nearby_markers(comp)
        component_markers.append((comp, markers))
    
    # Template is the one with most distinct marker colors
    template_comp, template_markers = max(component_markers, key=lambda x: len(x[1]))
    
    if not template_markers:
        return result
    
    # Get template reference point (top-left of bounding box)
    t_min_r, t_max_r, t_min_c, t_max_c = get_bbox(template_comp)
    
    # Build template pattern: relative positions of all marker colors
    template_pattern = {}
    for color, positions in template_markers.items():
        for r, c in positions:
            rel_r, rel_c = r - t_min_r, c - t_min_c
            template_pattern[(rel_r, rel_c)] = color
    
    # Also include frame positions
    frame_pattern = set()
    for r, c in template_comp:
        frame_pattern.add((r - t_min_r, c - t_min_c))
    
    # For each other component, complete it using the template
    for comp, markers in component_markers:
        if comp == template_comp:
            continue
        
        c_min_r, c_max_r, c_min_c, c_max_c = get_bbox(comp)
        
        # Fill in missing markers from template
        for (rel_r, rel_c), color in template_pattern.items():
            target_r = c_min_r + rel_r
            target_c = c_min_c + rel_c
            if 0 <= target_r < rows and 0 <= target_c < cols:
                # Only fill if not already a marker or if it's background/frame
                if result[target_r][target_c] == background or result[target_r][target_c] == frame_color:
                    result[target_r][target_c] = color
    
    return result


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['5b834496']
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        actual = transform(inp)
        
        match = actual == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected:
                print(row)
            print("Actual:")
            for row in actual:
                print(row)
            print()
