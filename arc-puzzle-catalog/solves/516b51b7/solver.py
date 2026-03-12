#!/usr/bin/env python3
"""
ARC-AGI Task 516b51b7 Solver

Rule: For each rectangular region filled with 1s, fill it with concentric layers:
- Outermost border stays 1
- Next layer inward becomes 2
- Next layer inward becomes 3
- etc.
"""

import json
import sys


def find_rectangles(grid):
    """Find all rectangular regions filled with 1s."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    visited = [[False] * cols for _ in range(rows)]
    rectangles = []
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited[i][j]:
                # Try to find the bounding rectangle
                rect = find_rectangle_at(grid, i, j, visited)
                if rect:
                    rectangles.append(rect)
    
    return rectangles


def find_rectangle_at(grid, start_r, start_c, visited):
    """Find the rectangular region starting at (start_r, start_c)."""
    rows = len(grid)
    cols = len(grid[0])
    
    # Find bounds of this rectangle
    min_r, max_r = start_r, start_r
    min_c, max_c = start_c, start_c
    
    # Expand to find full rectangle bounds
    # First, expand right and down to find bounding box
    while max_r + 1 < rows and grid[max_r + 1][start_c] == 1:
        max_r += 1
    while max_c + 1 < cols and grid[start_r][max_c + 1] == 1:
        max_c += 1
    
    # Verify it's actually a filled rectangle
    is_valid = True
    for i in range(min_r, max_r + 1):
        for j in range(min_c, max_c + 1):
            if grid[i][j] != 1:
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid:
        return None
    
    # Mark as visited
    for i in range(min_r, max_r + 1):
        for j in range(min_c, max_c + 1):
            visited[i][j] = True
    
    return (min_r, max_r, min_c, max_c)


def fill_rectangle_with_layers(grid, rect):
    """Fill a rectangle with concentric layers."""
    min_r, max_r, min_c, max_c = rect
    
    # Create output grid
    output = [row[:] for row in grid]
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    # For each layer (distance from border)
    max_depth = min(height, width) // 2
    
    for layer in range(max_depth):
        # The value for this layer (layer 0 stays 1, layer 1 becomes 2, etc.)
        value = layer + 1 if layer == 0 else layer + 1
        
        # Fill this layer
        for i in range(min_r + layer, max_r - layer + 1):
            for j in range(min_c + layer, max_c - layer + 1):
                # Check distance from border
                dist_from_border = min(
                    i - min_r,
                    max_r - i,
                    j - min_c,
                    max_c - j
                )
                
                # Assign value based on distance
                output[i][j] = dist_from_border + 1
    
    return output


def solve(grid):
    """Apply the transformation to fill rectangles with concentric layers."""
    output = [row[:] for row in grid]
    
    # Find all rectangular regions
    rectangles = find_rectangles(grid)
    
    # For each rectangle, fill with concentric layers
    for rect in rectangles:
        output = fill_rectangle_with_layers(output, rect)
    
    return output


def main():
    if len(sys.argv) > 1:
        task_file = sys.argv[1]
    else:
        task_file = "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/516b51b7.json"
    
    task_file = task_file.replace("~", "/Users/evanpieser")
    
    with open(task_file, 'r') as f:
        task = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        computed_output = solve(input_grid)
        
        if computed_output == expected_output:
            print(f"Training Example {idx + 1}: PASS")
        else:
            print(f"Training Example {idx + 1}: FAIL")
            all_pass = False
            # Print diff for debugging
            print(f"  Expected:\n{expected_output}")
            print(f"  Got:\n{computed_output}")
    
    if all_pass:
        print("\nAll training examples PASSED!")
    else:
        print("\nSome training examples FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
