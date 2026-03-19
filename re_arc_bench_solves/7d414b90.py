"""
ARC Puzzle 7d414b90 Solver

Pattern: Find scattered colored regions that are adjacent to a different "anchor" color.
Complete those scattered regions into filled rectangles using their bounding box.
"""

def transform(grid):
    import copy
    
    grid = [list(row) for row in grid]
    H, W = len(grid), len(grid[0])
    output = copy.deepcopy(grid)
    
    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all non-background pixels
    colored_pixels = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                color = grid[r][c]
                if color not in colored_pixels:
                    colored_pixels[color] = []
                colored_pixels[color].append((r, c))
    
    # Find connected components of non-background pixels (any non-bg color)
    visited = [[False]*W for _ in range(H)]
    
    def get_component(start_r, start_c):
        """BFS to find all connected non-bg pixels"""
        component = []
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= H or c < 0 or c >= W:
                continue
            if visited[r][c] or grid[r][c] == bg:
                continue
            visited[r][c] = True
            component.append((r, c, grid[r][c]))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                stack.append((r+dr, c+dc))
        return component
    
    components = []
    for r in range(H):
        for c in range(W):
            if not visited[r][c] and grid[r][c] != bg:
                comp = get_component(r, c)
                if comp:
                    components.append(comp)
    
    # For each component with multiple colors, fill bounding boxes
    for comp in components:
        colors_in_comp = set(color for r, c, color in comp)
        
        # Only process if component has multiple distinct colors (shape + anchor)
        if len(colors_in_comp) < 2:
            continue
        
        # For each color in component, find bounding box and fill
        for target_color in colors_in_comp:
            pixels_of_color = [(r, c) for r, c, color in comp if color == target_color]
            
            if len(pixels_of_color) < 2:
                # Single pixel anchors - don't fill
                continue
            
            # Get bounding box
            min_r = min(r for r, c in pixels_of_color)
            max_r = max(r for r, c in pixels_of_color)
            min_c = min(c for r, c in pixels_of_color)
            max_c = max(c for r, c in pixels_of_color)
            
            # Fill the bounding box with this color
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    # Only fill if currently bg or same color
                    if output[r][c] == bg or output[r][c] == target_color:
                        output[r][c] = target_color
    
    return [list(row) for row in output]


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['7d414b90']
    
    all_passed = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        all_passed = all_passed and match
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
            print()
    
    print(f"\nAll passed: {all_passed}")
