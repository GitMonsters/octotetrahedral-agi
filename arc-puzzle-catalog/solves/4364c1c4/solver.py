#!/usr/bin/env python3
"""
ARC Task 4364c1c4 Solver

Pattern: 
1. Find the background color (most common)
2. Find all connected components of non-background colors
3. Sort components by their top-left position (reading order)
4. For each component at index i (when sorted):
   - If i is even: shift all pixels of that component LEFT by 1 column
   - If i is odd: shift all pixels of that component RIGHT by 1 column
"""

from collections import deque

def find_connected_components(grid, bg_color):
    """Find all connected components of all non-bg colors, sorted by occurrence"""
    height = len(grid)
    width = len(grid[0])
    visited = set()
    components = []  # List of (top_row, left_col, color, positions)
    
    def bfs(start_r, start_c, color):
        comp = []
        queue = deque([(start_r, start_c)])
        visited.add((start_r, start_c))
        min_r = start_r
        min_c = start_c
        
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            min_r = min(min_r, r)
            min_c = min(min_c, c)
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                    if grid[nr][nc] == color:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        return comp, min_r, min_c
    
    for r in range(height):
        for c in range(width):
            if grid[r][c] != bg_color and (r, c) not in visited:
                color = grid[r][c]
                comp, min_r, min_c = bfs(r, c, color)
                components.append((min_r, min_c, color, comp))
    
    # Sort by occurrence (top-left reading order)
    components.sort(key=lambda x: (x[0], x[1]))
    return components


def solve(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    height = len(grid)
    width = len(grid[0])
    
    # Find the background color (most common color)
    color_counts = {}
    for row in grid:
        for val in row:
            color_counts[val] = color_counts.get(val, 0) + 1
    
    bg_color = max(color_counts, key=color_counts.get)
    
    # Find all connected components
    components = find_connected_components(grid, bg_color)
    
    # Create output grid filled with background color
    output = [[bg_color] * width for _ in range(height)]
    
    # For each connected component, shift horizontally based on its index
    for comp_idx, (_, _, color, positions) in enumerate(components):
        # Even index: shift left (-1), Odd index: shift right (+1)
        shift = -1 if comp_idx % 2 == 0 else 1
        
        # Find all positions of this component and shift them
        for r, c in positions:
            new_c = c + shift
            # Only place if within bounds
            if 0 <= new_c < width:
                output[r][new_c] = color
    
    return output


if __name__ == "__main__":
    import json
    import sys
    
    # Load task from JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/4364c1c4.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on all training examples
    print("Testing on training examples:")
    all_pass = True
    for idx, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        predicted_output = solve(input_grid)
        
        if predicted_output == expected_output:
            print(f"  Example {idx + 1}: PASS")
        else:
            print(f"  Example {idx + 1}: FAIL")
            all_pass = False
            # Print details for debugging
            print(f"    Expected shape: {len(expected_output)}x{len(expected_output[0])}")
            print(f"    Got shape: {len(predicted_output)}x{len(predicted_output[0])}")
            # Show first difference
            for r in range(len(expected_output)):
                for c in range(len(expected_output[0])):
                    if predicted_output[r][c] != expected_output[r][c]:
                        print(f"    First diff at ({r}, {c}): expected {expected_output[r][c]}, got {predicted_output[r][c]}")
                        break
    
    if all_pass:
        print("\nAll training examples PASSED!")
    else:
        print("\nSome training examples FAILED!")
        sys.exit(1)

