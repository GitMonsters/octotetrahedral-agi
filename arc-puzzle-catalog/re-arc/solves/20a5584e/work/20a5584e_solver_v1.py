#!/usr/bin/env python3

def solve(grid):
    """
    ARC-AGI Task 20a5584e Solution
    
    Algorithm:
    1. Find background color (most frequent)
    2. Find blue dots (color 1) 
    3. Find original pattern shapes (non-background, non-1 colors)
    4. Replicate pattern around each blue dot using the pattern [(-1,-1), (1,-1), (1,0)]
    """
    result = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    
    # Find background color (most frequent)
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    background_color = max(color_counts, key=color_counts.get)
    
    # Find blue dots (color 1)
    blue_dots = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 1:
                blue_dots.append((r, c))
    
    # Find original pattern colors (non-background, non-1)
    pattern_colors = set()
    for row in grid:
        for cell in row:
            if cell != background_color and cell != 1:
                pattern_colors.add(cell)
    
    if not pattern_colors or not blue_dots:
        return result
    
    # For each blue dot, replicate pattern using specific offsets
    pattern_offsets = [(-1, -1), (1, -1), (1, 0)]
    
    for blue_r, blue_c in blue_dots:
        # Apply pattern around this blue dot
        for i, (dr, dc) in enumerate(pattern_offsets):
            target_r, target_c = blue_r + dr, blue_c + dc
            
            if 0 <= target_r < height and 0 <= target_c < width:
                # Only change if it's background color (don't overwrite existing patterns)
                if grid[target_r][target_c] == background_color:
                    # Use the first pattern color found
                    pattern_color = list(pattern_colors)[0]
                    result[target_r][target_c] = pattern_color
    
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
        for r in range(len(expected_output)):
            for c in range(len(expected_output[0])):
                if expected_output[r][c] != actual_output[r][c]:
                    matches = False
                    break
            if not matches:
                break
        
        print(f"Training example {i+1}: {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            print("  Expected vs Actual differences:")
            diff_count = 0
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if expected_output[r][c] != actual_output[r][c]:
                        diff_count += 1
                        if diff_count <= 10:  # Show first 10 differences
                            print(f"    ({r},{c}): expected {expected_output[r][c]}, got {actual_output[r][c]}")
            if diff_count > 10:
                print(f"    ... and {diff_count - 10} more differences")

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
        for r in range(len(expected_output)):
            for c in range(len(expected_output[0])):
                if expected_output[r][c] != actual_output[r][c]:
                    matches = False
                    break
            if not matches:
                break
        
        print(f"Training example {i+1}: {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            print("  Expected vs Actual differences:")
            diff_count = 0
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if expected_output[r][c] != actual_output[r][c]:
                        diff_count += 1
                        if diff_count <= 10:  # Show first 10 differences
                            print(f"    ({r},{c}): expected {expected_output[r][c]}, got {actual_output[r][c]}")
            if diff_count > 10:
                print(f"    ... and {diff_count - 10} more differences")