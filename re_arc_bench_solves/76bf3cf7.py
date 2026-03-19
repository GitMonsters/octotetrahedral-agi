"""
Solver for ARC puzzle 76bf3cf7

Pattern: 
- Find template shape (connected blob of a specific color, typically 2)
- Template has anchor markers (other colors) inside/adjacent
- Copy template to each isolated marker position, anchored by matching color
- Erase original template
"""

def transform(grid):
    import copy
    
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Find background color (most frequent)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
    bg_color = max(color_counts, key=color_counts.get)
    
    # Find template color (forms connected blob, typically color 2)
    # Look for color that forms a significant connected component
    template_color = None
    for color in color_counts:
        if color == bg_color:
            continue
        # Check if this color forms a connected blob (more than just isolated pixels)
        positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        if len(positions) >= 3:
            # Check connectivity
            if is_connected_blob(positions):
                template_color = color
                break
    
    if template_color is None:
        return grid
    
    # Get template positions
    template_positions = set((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == template_color)
    
    # Get bounding box of template
    min_r = min(p[0] for p in template_positions)
    max_r = max(p[0] for p in template_positions)
    min_c = min(p[1] for p in template_positions)
    max_c = max(p[1] for p in template_positions)
    
    # Find anchor markers within/adjacent to template
    # Expand template region slightly to find adjacent markers
    template_region = set()
    for r, c in template_positions:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                template_region.add((r + dr, c + dc))
    
    # Find markers (non-bg, non-template) in template region
    anchor_markers = {}  # color -> (row, col) relative to template min corner
    for r, c in template_region:
        if 0 <= r < rows and 0 <= c < cols:
            color = grid[r][c]
            if color != bg_color and color != template_color:
                if color not in anchor_markers:
                    anchor_markers[color] = (r - min_r, c - min_c, r, c)  # relative pos + absolute pos
    
    # Find isolated markers outside template region
    isolated_markers = []  # (color, row, col)
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in template_region:
                color = grid[r][c]
                if color != bg_color and color != template_color:
                    isolated_markers.append((color, r, c))
    
    # Create output grid
    output = [[bg_color] * cols for _ in range(rows)]
    
    # Copy non-template, non-bg pixels (markers) to output
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg_color and grid[r][c] != template_color:
                output[r][c] = grid[r][c]
    
    # For each isolated marker, if its color has an anchor, draw template relative to it
    for marker_color, marker_r, marker_c in isolated_markers:
        if marker_color in anchor_markers:
            rel_r, rel_c, _, _ = anchor_markers[marker_color]
            # Calculate where template should be drawn
            # Template's anchor point was at (min_r + rel_r, min_c + rel_c)
            # Now we want anchor to be at (marker_r, marker_c)
            offset_r = marker_r - (min_r + rel_r)
            offset_c = marker_c - (min_c + rel_c)
            
            # Draw template at new position
            for tr, tc in template_positions:
                new_r = tr + offset_r
                new_c = tc + offset_c
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    output[new_r][new_c] = template_color
    
    return output


def is_connected_blob(positions):
    """Check if positions form a connected blob using flood fill."""
    if not positions:
        return False
    
    positions_set = set(positions)
    visited = set()
    
    def flood_fill(start):
        stack = [start]
        component = set()
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or (r, c) not in positions_set:
                continue
            visited.add((r, c))
            component.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in positions_set and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return component
    
    first = positions[0]
    component = flood_fill(first)
    
    # Connected if all positions are in one component
    return len(component) == len(positions_set)


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['76bf3cf7']
    
    all_passed = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        passed = result == expected
        all_passed = all_passed and passed
        
        print(f"Train {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
            print()
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
