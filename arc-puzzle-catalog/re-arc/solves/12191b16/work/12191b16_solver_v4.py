#!/usr/bin/env python3

def solve(grid):
    """
    ARC-AGI Task 12191b16 Solution - Direct Pattern Matching
    
    After analyzing the examples, the pattern is:
    1. Create symmetric rectangular frame extending the input pattern
    2. Each input row creates a symmetric pattern extending to the edges
    3. The pattern replicates the input values in specific positions
    """
    
    # Find background color (most frequent)
    flat_grid = [cell for row in grid for cell in row]
    bg_color = max(set(flat_grid), key=flat_grid.count)
    
    # Create output grid (start with copy of input)
    height, width = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    
    # Find non-background pixels and bounding box
    non_bg_pixels = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != bg_color:
                non_bg_pixels[(r, c)] = grid[r][c]
    
    if not non_bg_pixels:
        return output
    
    # Find bounding rectangle
    rows = [pos[0] for pos in non_bg_pixels.keys()]
    cols = [pos[1] for pos in non_bg_pixels.keys()]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    # Process each row that has non-background pixels
    for r in range(min_r, max_r + 1):
        input_row = grid[r]
        has_pattern = any(input_row[c] != bg_color for c in range(width))
        
        if has_pattern:
            # Create extended pattern for this row
            extended_pattern = create_exact_pattern_row(input_row, bg_color, width)
            output[r] = extended_pattern
            
            # Apply to symmetric row if different
            symmetric_r = height - 1 - r
            if 0 <= symmetric_r < height and symmetric_r != r:
                output[symmetric_r] = extended_pattern[:]
    
    return output


def create_exact_pattern_row(input_row, bg_color, width):
    """
    Create exact pattern row based on careful analysis of examples
    """
    # Find non-background positions and values
    non_bg_positions = []
    for c, value in enumerate(input_row):
        if value != bg_color:
            non_bg_positions.append((c, value))
    
    if not non_bg_positions:
        return input_row[:]
    
    # Create output row starting with background
    output_row = [bg_color] * width
    
    # Place the original input pattern first
    for c, value in enumerate(input_row):
        if value != bg_color:
            output_row[c] = value
    
    # Get the values and positions
    positions = [pos[0] for pos in non_bg_positions]
    values = [pos[1] for pos in non_bg_positions]
    
    # Find leftmost and rightmost non-background positions
    leftmost_pos = min(positions)
    rightmost_pos = max(positions)
    
    # Get leftmost and rightmost values
    leftmost_val = input_row[leftmost_pos]
    rightmost_val = input_row[rightmost_pos]
    
    # Strategy: Extend pattern symmetrically by analyzing the structure
    
    # Place values at edges (positions 1 and width-2 for odd positions)
    if width > 1:
        output_row[1] = leftmost_val
        output_row[width - 2] = rightmost_val
    
    # Fill intermediate odd positions based on pattern
    # This is the key insight from the examples
    
    # For each odd position between leftmost and rightmost
    for c in range(3, width - 1, 2):  # Skip positions 1 and width-2 already filled
        # Determine what value should go here based on pattern
        
        # Find the closest input pattern position
        closest_input_pos = min(positions, key=lambda p: abs(p - c))
        closest_input_val = input_row[closest_input_pos]
        
        # Use different strategy based on examples:
        # - If we're close to the original pattern, use values from there
        # - Otherwise use a default pattern value
        
        if c <= rightmost_pos:
            # We're in the original pattern region - use pattern values
            output_row[c] = closest_input_val
        else:
            # We're extending beyond - use pattern extension logic
            # Based on analysis, this seems to alternate or use specific values
            
            # Get all unique values for extension
            unique_vals = list(set(values))
            
            # Use position-based selection (this is empirical from examples)
            val_idx = ((c - rightmost_pos - 1) // 2) % len(unique_vals) 
            output_row[c] = unique_vals[val_idx]
    
    # Make pattern symmetric by mirroring
    for c in range(width // 2):
        mirror_c = width - 1 - c
        
        # If one side has a non-background value, mirror it
        if output_row[c] != bg_color:
            output_row[mirror_c] = output_row[c]
        elif output_row[mirror_c] != bg_color:
            output_row[c] = output_row[mirror_c]
    
    return output_row


# Test function
def test_solver():
    import json
    
    with open('/Users/evanpieser/apr12_tasks/12191b16.json', 'r') as f:
        task = json.load(f)
    
    print("Testing solver on training examples:")
    
    for i, pair in enumerate(task['train']):
        input_grid = pair['input']
        expected_output = pair['output']
        predicted_output = solve(input_grid)
        
        matches = predicted_output == expected_output
        print(f"Train {i+1}: {'PASS' if matches else 'FAIL'}")
        
        if not matches:
            # Show detailed comparison for first failing row
            for j in range(min(10, len(expected_output))):
                exp_row = expected_output[j]
                pred_row = predicted_output[j]
                if exp_row != pred_row:
                    print(f"  First mismatch at row {j}:")
                    print(f"    Expected:  {exp_row}")
                    print(f"    Predicted: {pred_row}")
                    
                    # Show position-by-position diff
                    diffs = []
                    for k in range(len(exp_row)):
                        if exp_row[k] != pred_row[k]:
                            diffs.append((k, exp_row[k], pred_row[k]))
                    print(f"    Diffs: {diffs[:5]}...")  # Show first 5 diffs
                    break
            print()

if __name__ == "__main__":
    test_solver()