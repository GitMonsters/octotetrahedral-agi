#!/usr/bin/env python3

def solve(grid):
    """
    ARC-AGI Task 12191b16 Solution
    
    Pattern: Transform input into a symmetric rectangular frame pattern where:
    1. Find the bounding box of non-background pixels
    2. Create a symmetric grid with the input pixels and their patterns
    3. The pattern extends symmetrically both horizontally and vertically
    4. Alternating rows of background color and pattern rows
    """
    
    # Find background color (most frequent)
    flat_grid = [cell for row in grid for cell in row]
    bg_color = max(set(flat_grid), key=flat_grid.count)
    
    # Find non-background pixels and their bounding box
    non_bg_pixels = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != bg_color:
                non_bg_pixels.append((r, c, grid[r][c]))
    
    if not non_bg_pixels:
        return [row[:] for row in grid]  # Return copy if no non-bg pixels
    
    # Find bounding rectangle
    rows = [p[0] for p in non_bg_pixels]
    cols = [p[1] for p in non_bg_pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    # Create output grid
    height, width = len(grid), len(grid[0])
    output = [[bg_color for _ in range(width)] for _ in range(height)]
    
    # Create the symmetric pattern
    # The pattern extends from the bounding box to create a symmetric frame
    
    # For each row that contained non-background pixels, create pattern rows
    pattern_rows = list(range(min_r, max_r + 1, 2))  # odd spacing like in examples
    
    # Extend pattern symmetrically above and below
    center_r = height // 2
    
    # Map the bounding box rows to symmetric positions
    for i, input_row in enumerate(range(min_r, max_r + 1)):
        if grid[input_row] != [bg_color] * width:  # Has non-background pixels
            # Create symmetric rows above and below center
            offset = input_row - min_r
            
            # Map to symmetric positions
            target_rows = []
            
            # Calculate symmetric positions around center
            pattern_start = 1  # Start from row 1 like in examples
            pattern_spacing = 2  # Every other row
            
            for j in range(len(pattern_rows)):
                target_r = pattern_start + j * pattern_spacing
                if target_r < height:
                    target_rows.append(target_r)
                
                # Add symmetric counterpart
                symmetric_r = height - 1 - target_r
                if symmetric_r >= 0 and symmetric_r != target_r:
                    target_rows.append(symmetric_r)
            
            # Fill the target rows with pattern
            for target_r in target_rows:
                if 0 <= target_r < height:
                    # Create symmetric pattern for this row
                    row_pattern = [bg_color] * width
                    
                    # Find the pattern based on the input row
                    input_pixels = {}
                    for c in range(width):
                        if input_row < len(grid) and c < len(grid[input_row]):
                            if grid[input_row][c] != bg_color:
                                input_pixels[c] = grid[input_row][c]
                    
                    # Create symmetric pattern
                    if input_pixels:
                        # Get unique colors and positions from input
                        unique_colors = list(set(input_pixels.values()))
                        unique_positions = sorted(input_pixels.keys())
                        
                        # Create alternating pattern extending to edges
                        for c in range(1, width, 2):  # Odd columns
                            if c in input_pixels:
                                row_pattern[c] = input_pixels[c]
                            elif unique_colors:
                                # Use pattern from the input colors
                                color_idx = ((c - 1) // 2) % len(unique_colors)
                                row_pattern[c] = unique_colors[color_idx]
                    
                    # Make it symmetric
                    for c in range(width):
                        mirror_c = width - 1 - c
                        if row_pattern[c] != bg_color:
                            row_pattern[mirror_c] = row_pattern[c]
                        elif row_pattern[mirror_c] != bg_color:
                            row_pattern[c] = row_pattern[mirror_c]
                    
                    output[target_r] = row_pattern
    
    # Actually, let me reimplement this more directly based on the observed pattern
    # The pattern seems to be creating a rectangular frame with the input colors
    
    # Reset and use direct approach
    output = [[bg_color for _ in range(width)] for _ in range(height)]
    
    # Create symmetric frame pattern
    # Find center and create alternating pattern rows
    for r in range(1, height, 2):  # Every other row starting from 1
        if r < height:
            row_pattern = [bg_color] * width
            
            # Create symmetric pattern based on input
            # Place non-background colors in alternating positions
            colors_used = [pixel[2] for pixel in non_bg_pixels]
            unique_colors = list(set(colors_used))
            
            if unique_colors:
                # Create alternating pattern
                for c in range(1, width, 2):  # Odd positions
                    color_idx = (c - 1) // 2 % len(unique_colors)
                    row_pattern[c] = unique_colors[color_idx]
                
                # Make symmetric
                for c in range(width // 2):
                    mirror_c = width - 1 - c
                    if row_pattern[c] != bg_color or row_pattern[mirror_c] != bg_color:
                        color = row_pattern[c] if row_pattern[c] != bg_color else row_pattern[mirror_c]
                        row_pattern[c] = color
                        row_pattern[mirror_c] = color
            
            output[r] = row_pattern
    
    return output


# Test function
def test_solver():
    import json
    
    # Load test cases
    with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
        task = json.load(f)
    
    print("Testing solver on training examples:")
    
    for i, pair in enumerate(task['train']):
        input_grid = pair['input']
        expected_output = pair['output']
        predicted_output = solve(input_grid)
        
        # Check if prediction matches expected
        matches = predicted_output == expected_output
        print(f"Train {i+1}: {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            print("Expected:")
            for row in expected_output[:5]:  # Show first 5 rows
                print("  ", row)
            print("Got:")
            for row in predicted_output[:5]:  # Show first 5 rows
                print("  ", row)
            print()

if __name__ == "__main__":
    test_solver()