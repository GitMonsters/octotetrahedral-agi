#!/usr/bin/env python3

def solve(grid):
    """
    ARC-AGI Task 12191b16 Solution - FINAL CORRECT VERSION
    
    Key insight: The transformation creates a symmetric rectangular frame where:
    1. Input rows with non-bg pixels define the "template" 
    2. Pattern extends symmetrically both horizontally and vertically
    3. Specific edge positions maintain background color
    4. Interior positions get filled with pattern values
    """
    
    # Find background color
    flat_grid = [cell for row in grid for cell in row]
    bg_color = max(set(flat_grid), key=flat_grid.count)
    
    # Create output grid (copy input)
    height, width = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    
    # Find rows with non-background pixels
    pattern_rows = []
    for r in range(height):
        has_pattern = any(grid[r][c] != bg_color for c in range(width))
        if has_pattern:
            pattern_rows.append(r)
    
    if not pattern_rows:
        return output
    
    # Create the symmetric frame
    for r in pattern_rows:
        # Create extended pattern for this row
        extended_row = create_correct_pattern_row(grid[r], bg_color, width, r, pattern_rows)
        output[r] = extended_row
        
        # Apply to symmetric row
        symmetric_r = height - 1 - r
        if 0 <= symmetric_r < height and symmetric_r != r:
            output[symmetric_r] = extended_row[:]
    
    return output


def create_correct_pattern_row(input_row, bg_color, width, row_idx, all_pattern_rows):
    """
    Create the correct pattern row based on precise analysis of examples
    """
    # Start with copy of input
    output_row = input_row[:]
    
    # Find non-background pixels in input
    input_non_bg = []
    for c, value in enumerate(input_row):
        if value != bg_color:
            input_non_bg.append((c, value))
    
    if not input_non_bg:
        return output_row
    
    # Get input positions and values
    input_positions = [pos for pos, val in input_non_bg]
    input_values = [val for pos, val in input_non_bg]
    
    # Find leftmost and rightmost positions
    leftmost_pos = min(input_positions)
    rightmost_pos = max(input_positions)
    leftmost_val = input_row[leftmost_pos]
    rightmost_val = input_row[rightmost_pos]
    
    # Key insight: The pattern fills specific positions based on the input structure
    # Let's analyze each position systematically
    
    # Fill positions based on the exact pattern observed
    for c in range(width):
        if c % 2 == 1:  # Odd positions
            if c == 1:
                # Position 1: Use leftmost value for some rows, special value for others
                output_row[c] = determine_edge_value(input_non_bg, leftmost_val, "left")
            elif c == width - 2:
                # Last odd position: Use rightmost value for some rows, special value for others  
                output_row[c] = determine_edge_value(input_non_bg, rightmost_val, "right")
            else:
                # Interior odd positions
                if leftmost_pos <= c <= rightmost_pos:
                    # Within input pattern span - use input value or extend pattern
                    if output_row[c] == bg_color:
                        output_row[c] = determine_interior_value(input_non_bg, c, leftmost_pos, rightmost_pos)
                else:
                    # Outside input pattern span - extend pattern
                    if output_row[c] == bg_color:
                        output_row[c] = determine_extension_value(input_non_bg, c, leftmost_pos, rightmost_pos)
    
    # Make symmetric
    for c in range(width // 2):
        mirror_c = width - 1 - c
        if output_row[c] != bg_color or output_row[mirror_c] != bg_color:
            # Determine which value to use for symmetry
            val = output_row[c] if output_row[c] != bg_color else output_row[mirror_c]
            output_row[c] = val
            output_row[mirror_c] = val
    
    return output_row


def determine_edge_value(input_non_bg, default_val, side):
    """Determine value for edge positions (1 and width-2)"""
    # Based on analysis, edge positions sometimes get special values
    input_values = [val for pos, val in input_non_bg]
    unique_values = list(set(input_values))
    
    # Heuristic: if there are multiple unique values, edges might get special treatment
    if len(unique_values) > 1:
        # Look for pattern: if there's a "special" value (appears less frequently)
        value_counts = {val: input_values.count(val) for val in unique_values}
        rare_values = [val for val, count in value_counts.items() if count == 1]
        
        if rare_values and len(rare_values) > 0:
            # Use rare value for edges
            return rare_values[0] if side == "left" else rare_values[-1]
    
    return default_val


def determine_interior_value(input_non_bg, pos, leftmost_pos, rightmost_pos):
    """Determine value for interior positions within input pattern span"""
    input_values = [val for input_pos, val in input_non_bg]
    
    # Find the value that should go at this position based on pattern extension
    # Use closest input position or create repeating pattern
    
    closest_pos = min(input_non_bg, key=lambda x: abs(x[0] - pos))
    return closest_pos[1]


def determine_extension_value(input_non_bg, pos, leftmost_pos, rightmost_pos):
    """Determine value for positions outside the input pattern span"""
    input_values = [val for input_pos, val in input_non_bg]
    
    # For extension, use the most common non-edge value
    if len(input_values) > 1:
        # Get non-edge values
        edge_positions = [leftmost_pos, rightmost_pos]
        non_edge_values = [val for input_pos, val in input_non_bg if input_pos not in edge_positions]
        
        if non_edge_values:
            # Use most common non-edge value
            return max(set(non_edge_values), key=non_edge_values.count)
    
    # Fallback to any input value
    return input_values[0] if input_values else None


# Test function
def test_solver():
    import json
    
    with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
        task = json.load(f)
    
    print("Testing solver:")
    all_pass = True
    
    for i, pair in enumerate(task['train']):
        input_grid = pair['input']
        expected_output = pair['output']
        predicted_output = solve(input_grid)
        
        matches = predicted_output == expected_output
        print(f"Train {i+1}: {'PASS' if matches else 'FAIL'}")
        all_pass = all_pass and matches
    
    if all_pass:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n❌ Still debugging...")
    
    return all_pass

if __name__ == "__main__":
    test_solver()