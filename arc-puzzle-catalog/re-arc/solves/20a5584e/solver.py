#!/usr/bin/env python3

def solve(grid):
    """
    ARC-AGI solver for task 20a5584e
    
    Rule: For each single-cell marker, place any multi-cell pattern
    at offset (pattern_offset) from the marker, only on background cells.
    """
    
    # Create a copy of the grid for output
    result = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    
    # Find background color (most frequent)
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    background_color = max(color_counts, key=color_counts.get)
    
    # Find connected components for each color
    def find_connected_components(target_color):
        visited = [[False] * width for _ in range(height)]
        components = []
        
        def dfs(r, c, component):
            if r < 0 or r >= height or c < 0 or c >= width or visited[r][c] or grid[r][c] != target_color:
                return
            visited[r][c] = True
            component.append((r, c))
            dfs(r+1, c, component)
            dfs(r-1, c, component)  
            dfs(r, c+1, component)
            dfs(r, c-1, component)
        
        for r in range(height):
            for c in range(width):
                if not visited[r][c] and grid[r][c] == target_color:
                    component = []
                    dfs(r, c, component)
                    if component:
                        components.append(component)
        
        return components
    
    # Identify all colors
    all_colors = set()
    for row in grid:
        all_colors.update(row)
    non_bg_colors = all_colors - {background_color}
    
    # Separate markers (single cells) and patterns (multi-cell)
    markers = []
    patterns = []
    
    for color in non_bg_colors:
        components = find_connected_components(color)
        for component in components:
            if len(component) == 1:
                markers.append((color, component[0]))
            else:
                patterns.append((color, component))
    
    # For each pattern, replicate it around each marker with offset (1, -1)
    for pattern_color, pattern_positions in patterns:
        # Get pattern shape relative to its top-left position
        min_r = min(r for r, c in pattern_positions)
        min_c = min(c for r, c in pattern_positions)
        pattern_shape = [(r - min_r, c - min_c) for r, c in pattern_positions]
        
        # Apply this pattern around each marker with offset
        for marker_color, marker_pos in markers:
            marker_r, marker_c = marker_pos
            
            # Place pattern with offset (1, -1) from marker
            pattern_start_r = marker_r + 1
            pattern_start_c = marker_c - 1
            
            for dr, dc in pattern_shape:
                new_r, new_c = pattern_start_r + dr, pattern_start_c + dc
                
                # Only place if within bounds and on background
                if (0 <= new_r < height and 
                    0 <= new_c < width and 
                    result[new_r][new_c] == background_color):
                    result[new_r][new_c] = pattern_color
    
    return result

# Test the solver on training examples
if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
        task = json.load(f)
    
    print("Testing solver on training examples...")
    
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        actual_output = solve(input_grid)
        
        # Compare outputs
        matches = True
        differences = []
        for r in range(len(expected_output)):
            for c in range(len(expected_output[0])):
                if expected_output[r][c] != actual_output[r][c]:
                    matches = False
                    differences.append((r, c, expected_output[r][c], actual_output[r][c]))
        
        print(f"Training example {i+1}: {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            print(f"  Total differences: {len(differences)}")
            for j, (r, c, exp, got) in enumerate(differences[:10]):
                print(f"    ({r},{c}): expected {exp}, got {got}")
            if len(differences) > 10:
                print(f"    ... and {len(differences) - 10} more differences")