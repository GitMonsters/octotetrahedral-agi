#!/usr/bin/env python3

import json
import sys


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Transform rule:
    1. Find the vertical line of 5s (appears in one column)
    2. Fill left side (cols 0 to line_col-1) with 8s
    3. Keep the vertical line of 5s
    4. Fill right side with 6s that decay as you go down (diagonal decay pattern)
    5. Create a staircase pattern of 8s that extends below the line
    """
    
    # Find the vertical line of 5s
    line_col = None
    line_start_row = None
    line_end_row = None
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 5:
                if line_col is None:
                    line_col = c
                    line_start_row = r
                line_end_row = r
    
    if line_col is None:
        # No line found, return grid as is
        return [row[:] for row in grid]
    
    line_length = line_end_row - line_start_row + 1
    
    # Create output grid
    output = [row[:] for row in grid]
    
    # Fill left side with 8s and add 6s on the right
    for r in range(line_length):
        # Fill left side with 8s (cols 0 to line_col-1)
        for c in range(line_col):
            output[r][c] = 8
        
        # Add 6s on the right side with decay pattern
        # The 6s width decreases as we go down
        # Pattern: row 0 gets max width, then it alternates/decreases
        
        # Calculate how many 6s to place in this row
        # For paired rows: first of pair gets more 6s, second gets fewer
        pair_index = r // 2
        is_first_of_pair = (r % 2 == 0)
        
        # Number of 6s decreases with pair_index
        if is_first_of_pair:
            num_sixes = max(0, line_length - pair_index - 1)
        else:
            num_sixes = max(0, line_length - pair_index - 2)
        
        # Place the 6s starting from line_col+1
        for i in range(num_sixes):
            if line_col + 1 + i < len(output[0]):
                output[r][line_col + 1 + i] = 6
    
    # Create staircase pattern of 8s below the line
    # The 8s width decreases as we go down
    for r in range(line_end_row + 1, len(output)):
        steps_below = r - line_end_row
        # Width of 8s decreases: starts at line_col, decreases by (steps_below // 2 + 1)
        eight_width = max(1, line_col - (steps_below // 2))
        
        for c in range(eight_width):
            output[r][c] = 8
    
    return output


if __name__ == '__main__':
    # Load task JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5207a7b5.json'
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        actual_output = solve(input_grid)
        
        # Compare
        match = actual_output == expected_output
        
        if match:
            print(f"Training example {idx + 1}: PASS")
        else:
            print(f"Training example {idx + 1}: FAIL")
            all_pass = False
            
            # Show differences
            for r in range(len(actual_output)):
                for c in range(len(actual_output[0])):
                    if actual_output[r][c] != expected_output[r][c]:
                        print(f"  Row {r}, Col {c}: expected {expected_output[r][c]}, got {actual_output[r][c]}")
    
    if all_pass:
        print("\n✓ All training examples PASS")
    else:
        print("\n✗ Some training examples FAIL")
        sys.exit(1)
