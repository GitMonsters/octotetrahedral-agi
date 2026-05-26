#!/usr/bin/env python3

def solve(grid):
    """
    ARC-AGI Task 12191b16 Solution - FINAL
    
    Pattern: 
    1. Find non-background pixels to define bounding rectangle
    2. Create symmetric extension of the pattern both horizontally and vertically
    3. Each input row with pattern gets mapped to corresponding rows in output
    4. The pattern extends symmetrically to create a rectangular frame
    """
    
    # Find background color (most frequent)
    flat_grid = [cell for row in grid for cell in row]
    bg_color = max(set(flat_grid), key=flat_grid.count)
    
    # Find non-background pixels and bounding box
    non_bg_pixels = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != bg_color:
                non_bg_pixels[(r, c)] = grid[r][c]
    
    if not non_bg_pixels:
        return [row[:] for row in grid]
    
    # Find bounding rectangle
    rows = [pos[0] for pos in non_bg_pixels.keys()]
    cols = [pos[1] for pos in non_bg_pixels.keys()]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    # Create output grid (start with copy of input)
    height, width = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    
    # Find the center of the grid for symmetric extension
    center_r = height // 2
    center_c = width // 2
    
    # Process each row that has non-background pixels
    input_pattern_rows = []
    for r in range(min_r, max_r + 1):
        has_pattern = any(grid[r][c] != bg_color for c in range(width))
        if has_pattern:
            input_pattern_rows.append(r)
    
    # Map input rows to symmetric output regions
    for input_row_idx in input_pattern_rows:
        # Get the pattern from this input row
        input_row = grid[input_row_idx]
        
        # Create extended symmetric pattern for this row
        extended_pattern = create_extended_pattern_row(input_row, bg_color, width, min_c, max_c)
        
        # Apply this pattern to the corresponding output row
        output[input_row_idx] = extended_pattern
        
        # Also apply to symmetric rows (vertically symmetric)
        symmetric_row = height - 1 - input_row_idx
        if 0 <= symmetric_row < height and symmetric_row != input_row_idx:
            output[symmetric_row] = extended_pattern[:]
    
    return output


def create_extended_pattern_row(input_row, bg_color, width, min_c, max_c):
    """
    Create extended symmetric pattern row based on input row
    """
    # Start with copy of input
    extended_row = input_row[:]
    
    # Find non-background positions in input
    non_bg_positions = []
    for c, value in enumerate(input_row):
        if value != bg_color:
            non_bg_positions.append((c, value))
    
    if not non_bg_positions:
        return extended_row
    
    # Find the span of the input pattern
    pattern_cols = [pos[0] for pos in non_bg_positions]
    pattern_min_c = min(pattern_cols)
    pattern_max_c = max(pattern_cols)
    
    # Create the extended pattern by filling in positions symmetrically
    # Based on the observed pattern, we need to:
    # 1. Extend the pattern to cover the full width symmetrically
    # 2. Use values from the input pattern to fill intermediate positions
    
    # Get unique values from the pattern
    pattern_values = [pos[1] for pos in non_bg_positions]
    unique_values = list(set(pattern_values))
    
    # Create symmetric pattern around the center
    center_col = width // 2
    
    # Extend pattern symmetrically
    for c in range(width):
        if extended_row[c] == bg_color:  # Only fill background positions
            # Calculate distance from center for symmetric placement
            dist_from_center = abs(c - center_col)
            
            # Determine position within pattern region
            # This is a heuristic based on the observed patterns
            mirror_c = width - 1 - c
            
            # Check if we should place a pattern value here based on observed structure
            # Pattern seems to extend to create alternating values in certain positions
            
            # Use position-based heuristic
            if c % 2 == 1:  # Odd positions seem to get pattern values
                # Determine which value to use based on distance/position
                if len(unique_values) > 0:
                    # Use different strategies based on distance from edges
                    dist_from_left = c
                    dist_from_right = width - 1 - c
                    min_dist_to_edge = min(dist_from_left, dist_from_right)
                    
                    # Different values for different distances (approximation)
                    if min_dist_to_edge <= 1:  # Near edges
                        # Use edge values from input pattern if available
                        if non_bg_positions:
                            edge_values = [non_bg_positions[0][1], non_bg_positions[-1][1]]
                            extended_row[c] = edge_values[min_dist_to_edge % len(edge_values)]
                    else:
                        # Use pattern values for interior
                        value_idx = (dist_from_center - 2) % len(unique_values)
                        extended_row[c] = unique_values[value_idx]
    
    # Make the pattern exactly symmetric
    for c in range(width // 2):
        mirror_c = width - 1 - c
        if extended_row[c] != bg_color or extended_row[mirror_c] != bg_color:
            # Make both sides the same (prefer non-background)
            value = extended_row[c] if extended_row[c] != bg_color else extended_row[mirror_c]
            extended_row[c] = value
            extended_row[mirror_c] = value
    
    return extended_row


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
            print("First 5 rows comparison:")
            for j in range(min(5, len(expected_output))):
                exp_row = expected_output[j]
                pred_row = predicted_output[j]
                match_row = exp_row == pred_row
                print(f"  Row {j} {'✓' if match_row else '✗'}")
                if not match_row:
                    print(f"    Expected:  {exp_row}")
                    print(f"    Predicted: {pred_row}")
            print()
    
    return all_pass

if __name__ == "__main__":
    test_solver()