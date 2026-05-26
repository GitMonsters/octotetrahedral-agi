def solve(grid):
    """
    ARC-AGI 50e65b9f solver - final corrected version.
    
    Pattern: For each connected component of any non-background color:
    1. Find the least frequent non-background color (fill_color)
    2. Create rectangular extensions:
       - Always try RIGHT extension
       - For larger rectangular components, also try DOWN extension
       - The combination creates larger rectangular filled areas
    """
    import copy
    
    result = copy.deepcopy(grid)
    
    # Find all non-background colors and their counts
    color_counts = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 8:  # 8 is background
                color = grid[r][c]
                color_counts[color] = color_counts.get(color, 0) + 1
    
    # If only one color or no colors, return as-is  
    if len(color_counts) <= 1:
        return result
    
    # Find the least frequent color - this becomes our fill color
    fill_color = min(color_counts, key=color_counts.get)
    
    # Find connected components
    visited = set()
    
    def flood_fill(start_r, start_c, target_color):
        component = []
        stack = [(start_r, start_c)]
        temp_visited = set()
        
        while stack:
            r, c = stack.pop()
            if (r, c) in temp_visited or r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                continue
            if grid[r][c] != target_color:
                continue
                
            temp_visited.add((r, c))
            component.append((r, c))
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                stack.append((r+dr, c+dc))
        
        return component
    
    # Find all components  
    components = []
    visited = set()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 8 and (r, c) not in visited:
                comp = flood_fill(r, c, grid[r][c])
                if comp:
                    visited.update(comp)
                    # Don't extend components that are already the fill color
                    if grid[comp[0][0]][comp[0][1]] != fill_color:
                        components.append(comp)
    
    # For each component, create extensions
    for comp in components:
        # Get bounding box
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)
        
        width = max_c - min_c + 1
        height = max_r - min_r + 1
        
        # Always try right extension
        extensions_to_try = [(0, width)]
        
        # For larger rectangular areas, also try down extension
        if width >= 2 and height >= 2 and len(comp) >= 4:
            extensions_to_try.append((height, 0))
        
        for dr, dc in extensions_to_try:
            # Check bounds for this extension
            new_min_r = min_r + dr
            new_max_r = max_r + dr
            new_min_c = min_c + dc  
            new_max_c = max_c + dc
            
            if (new_min_r >= 0 and new_max_r < len(grid) and
                new_min_c >= 0 and new_max_c < len(grid[0])):
                
                # Check if all target positions are background in ORIGINAL grid
                can_extend = True
                for r, c in comp:
                    new_r, new_c = r + dr, c + dc
                    if grid[new_r][new_c] != 8:  # Check original, not result
                        can_extend = False
                        break
                
                if can_extend:
                    # Fill the extension with fill_color
                    for r, c in comp:
                        new_r, new_c = r + dr, c + dc
                        result[new_r][new_c] = fill_color
    
    return result


if __name__ == "__main__":
    # Test on training data
    import json
    
    with open('/Users/evanpieser/apr12_tasks/50e65b9f.json', 'r') as f:
        task_data = json.load(f)
    
    print("Testing solver on training data:")
    all_passed = True
    
    for i, pair in enumerate(task_data['train']):
        input_grid = pair['input']
        expected_output = pair['output']
        predicted_output = solve(input_grid)
        
        matches = predicted_output == expected_output
        print(f"Train {i+1}: {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            all_passed = False
            
            # Count differences by change type
            background_to_fill = 0
            fill_to_background = 0
            other_changes = 0
            
            for r in range(len(input_grid)):
                for c in range(len(input_grid[0])):
                    if predicted_output[r][c] != expected_output[r][c]:
                        if input_grid[r][c] == 8 and expected_output[r][c] != 8:
                            # Should have filled background
                            background_to_fill += 1
                        elif input_grid[r][c] == 8 and predicted_output[r][c] != 8:
                            # Incorrectly filled background  
                            fill_to_background += 1
                        else:
                            other_changes += 1
            
            print(f"  Missing fills: {background_to_fill}")
            print(f"  Extra fills: {fill_to_background}")
            print(f"  Other changes: {other_changes}")
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")