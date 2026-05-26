def solve(grid):
    """
    Refined ARC-AGI 50e65b9f solver based on pattern analysis.
    
    Pattern: For each connected component, create extensions using the least 
    frequent non-background color as fill. Extensions are prioritized:
    1. Right extension (most common)
    2. Down extension (when right blocked/unavailable)
    3. Diagonal extension (when both right and down blocked/unavailable)
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
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                continue
            if grid[r][c] != target_color:
                continue
                
            visited.add((r, c))
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
                    components.append(comp)
    
    # For each component, try to create extensions
    for comp in components:
        # Get bounding box
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)
        
        width = max_c - min_c + 1
        height = max_r - min_r + 1
        
        # Try extensions in order of priority
        extensions = [
            (0, width),      # right (highest priority)
            (height, 0),     # down (medium priority)  
            (height, width)  # diagonal down-right (lowest priority)
        ]
        
        extension_created = False
        
        for dr, dc in extensions:
            # Skip if already created an extension
            if extension_created:
                break
                
            # Check bounds for this extension
            new_min_r = min_r + dr
            new_max_r = max_r + dr
            new_min_c = min_c + dc  
            new_max_c = max_c + dc
            
            if (new_min_r >= 0 and new_max_r < len(grid) and
                new_min_c >= 0 and new_max_c < len(grid[0])):
                
                # Check if all target positions are background
                can_extend = True
                for r, c in comp:
                    new_r, new_c = r + dr, c + dc
                    if result[new_r][new_c] != 8:
                        can_extend = False
                        break
                
                if can_extend:
                    # Fill the extension with fill_color
                    for r, c in comp:
                        new_r, new_c = r + dr, c + dc
                        result[new_r][new_c] = fill_color
                    extension_created = True
    
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
            print(f"  Expected shape: {len(expected_output)}x{len(expected_output[0])}")
            print(f"  Predicted shape: {len(predicted_output)}x{len(predicted_output[0])}")
            
            # Show first few differences
            diff_count = 0
            for r in range(len(input_grid)):
                for c in range(len(input_grid[0])):
                    if predicted_output[r][c] != expected_output[r][c]:
                        if diff_count < 10:
                            print(f"  Diff at ({r},{c}): predicted {predicted_output[r][c]}, expected {expected_output[r][c]}")
                        diff_count += 1
            print(f"  Total differences: {diff_count}")
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")