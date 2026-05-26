#!/usr/bin/env python3

def solve(grid):
    """
    ARC-AGI Task 12191b16 Solution - Corrected
    
    Pattern analysis:
    1. Find non-background pixels to define the pattern region
    2. Create a symmetric rectangular grid where each input row with non-bg pixels 
       gets mapped to symmetric pattern rows in the output
    3. The pattern extends the input structure symmetrically across the full width
    """
    
    # Find background color (most frequent)
    flat_grid = [cell for row in grid for cell in row]
    bg_color = max(set(flat_grid), key=flat_grid.count)
    
    # Find non-background pixels and their bounding box
    non_bg_pixels = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != bg_color:
                non_bg_pixels[(r, c)] = grid[r][c]
    
    if not non_bg_pixels:
        return [row[:] for row in grid]  # Return copy if no non-bg pixels
    
    # Find bounding rectangle
    rows = [pos[0] for pos in non_bg_pixels.keys()]
    cols = [pos[1] for pos in non_bg_pixels.keys()]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    # Create output grid
    height, width = len(grid), len(grid[0])
    output = [[bg_color for _ in range(width)] for _ in range(height)]
    
    # Map each input row with non-bg pixels to symmetric pattern rows
    input_rows_with_patterns = []
    for r in range(min_r, max_r + 1):
        row_has_pattern = False
        for c in range(width):
            if r < height and c < width and grid[r][c] != bg_color:
                row_has_pattern = True
                break
        if row_has_pattern:
            input_rows_with_patterns.append(r)
    
    # Create symmetric pattern rows - alternating rows starting from 1
    pattern_row_indices = []
    for i in range(1, height, 2):  # 1, 3, 5, 7, ...
        pattern_row_indices.append(i)
    
    # Map input pattern rows to output pattern rows symmetrically
    center_output_row = len(pattern_row_indices) // 2
    center_input_row = len(input_rows_with_patterns) // 2
    
    for i, input_row_idx in enumerate(input_rows_with_patterns):
        # Calculate symmetric mapping
        offset_from_center = i - center_input_row
        
        # Map to output rows symmetrically around center
        output_rows_for_this_input = []
        
        if center_output_row + offset_from_center < len(pattern_row_indices):
            target_row = pattern_row_indices[center_output_row + offset_from_center]
            if 0 <= target_row < height:
                output_rows_for_this_input.append(target_row)
        
        # Add symmetric counterpart
        if center_output_row - offset_from_center < len(pattern_row_indices) and center_output_row - offset_from_center >= 0:
            symmetric_row = pattern_row_indices[center_output_row - offset_from_center]
            if 0 <= symmetric_row < height and symmetric_row not in output_rows_for_this_input:
                output_rows_for_this_input.append(symmetric_row)
        
        # Create pattern for these output rows
        for output_row_idx in output_rows_for_this_input:
            if 0 <= output_row_idx < height:
                # Create the pattern for this row based on input row
                output[output_row_idx] = create_symmetric_pattern_row(
                    grid[input_row_idx], bg_color, width, min_c, max_c
                )
    
    return output


def create_symmetric_pattern_row(input_row, bg_color, target_width, min_c, max_c):
    """Create a symmetric pattern row based on the input row"""
    
    # Extract non-background pixels and their positions
    non_bg_positions = []
    for c, value in enumerate(input_row):
        if value != bg_color:
            non_bg_positions.append((c, value))
    
    if not non_bg_positions:
        return [bg_color] * target_width
    
    # Create the pattern row
    pattern_row = [bg_color] * target_width
    
    # Strategy: replicate the input pattern structure but extend it symmetrically
    # Find the range of the pattern in the input
    pattern_cols = [pos[0] for pos in non_bg_positions]
    pattern_min_c, pattern_max_c = min(pattern_cols), max(pattern_cols)
    pattern_span = pattern_max_c - pattern_min_c + 1
    
    # Map to output positions, centering and extending
    center_col = target_width // 2
    
    # Place the core pattern around the center
    for pos, value in non_bg_positions:
        # Calculate relative position in input pattern
        rel_pos = pos - pattern_min_c
        
        # Map to output position around center
        output_pos = center_col - pattern_span // 2 + rel_pos
        
        if 0 <= output_pos < target_width:
            pattern_row[output_pos] = value
            
        # Also place symmetric counterpart
        symmetric_pos = target_width - 1 - output_pos
        if 0 <= symmetric_pos < target_width:
            pattern_row[symmetric_pos] = value
    
    # Extend pattern to fill alternating positions symmetrically
    # This is complex - let me try a different approach based on observed patterns
    
    # Reset and use the observation that patterns seem to extend across odd positions
    pattern_row = [bg_color] * target_width
    
    # Get unique values from input
    unique_values = list(set(pos[1] for pos in non_bg_positions))
    
    # Fill odd positions with pattern, making it symmetric
    for c in range(1, target_width, 2):  # odd positions: 1, 3, 5, ...
        # Determine which value to place based on distance from edges
        dist_from_left = c
        dist_from_right = target_width - 1 - c
        min_dist_to_edge = min(dist_from_left, dist_from_right)
        
        # Use different values for different distances (rough approximation)
        if len(unique_values) > 0:
            value_idx = min_dist_to_edge % len(unique_values)
            pattern_row[c] = unique_values[value_idx]
    
    # Make completely symmetric
    for c in range(target_width // 2):
        mirror_c = target_width - 1 - c
        if pattern_row[c] != bg_color or pattern_row[mirror_c] != bg_color:
            # Take the non-background value
            value = pattern_row[c] if pattern_row[c] != bg_color else pattern_row[mirror_c]
            pattern_row[c] = value
            pattern_row[mirror_c] = value
    
    return pattern_row


# Test function
def test_solver():
    import json
    
    # Load test cases
    with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
        task = json.load(f)
    
    print("Testing solver on training examples:")
    
    all_pass = True
    for i, pair in enumerate(task['train']):
        input_grid = pair['input']
        expected_output = pair['output']
        predicted_output = solve(input_grid)
        
        # Check if prediction matches expected
        matches = predicted_output == expected_output
        print(f"Train {i+1}: {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            all_pass = False
            print("Expected first few rows:")
            for j, row in enumerate(expected_output[:5]):
                print(f"  {j}: {row}")
            print("Got first few rows:")
            for j, row in enumerate(predicted_output[:5]):
                print(f"  {j}: {row}")
            print()
    
    return all_pass

if __name__ == "__main__":
    test_solver()
