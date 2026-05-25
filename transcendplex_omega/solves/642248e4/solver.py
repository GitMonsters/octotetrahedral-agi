import json
import sys
from copy import deepcopy


def solve(grid):
    """
    Solve ARC puzzle 642248e4.
    
    Rule: For each cell with value 1, fill the adjacent cell in the direction
    towards the nearest border. The fill color is the color of that border.
    
    Borders: uniform color rows/columns on edges
    - If a border row/column exists at top/bottom/left/right, that's the border color
    - For each 1: determine which border it's closest to and fill adjacent cell
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    
    result = deepcopy(grid)
    
    # Identify border colors
    top_color = None
    bottom_color = None
    left_color = None
    right_color = None
    
    # Check if top row is uniform
    if h > 0 and all(grid[0][j] == grid[0][0] for j in range(w)):
        top_color = grid[0][0]
    
    # Check if bottom row is uniform
    if h > 0 and all(grid[h-1][j] == grid[h-1][0] for j in range(w)):
        bottom_color = grid[h-1][0]
    
    # Check if left column is uniform
    if w > 0 and all(grid[i][0] == grid[0][0] for i in range(h)):
        left_color = grid[0][0]
    
    # Check if right column is uniform
    if w > 0 and all(grid[i][w-1] == grid[0][w-1] for i in range(h)):
        right_color = grid[0][w-1]
    
    # Process each cell with value 1
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 1:
                # Determine closest border
                dist_to_top = i
                dist_to_bottom = h - 1 - i
                dist_to_left = j
                dist_to_right = w - 1 - j
                
                # If there are horizontal borders, use them
                if top_color is not None or bottom_color is not None:
                    if dist_to_top <= dist_to_bottom:
                        # Closer to top
                        if top_color is not None and i > 0:
                            result[i-1][j] = top_color
                    else:
                        # Closer to bottom
                        if bottom_color is not None and i < h - 1:
                            result[i+1][j] = bottom_color
                # Otherwise use vertical borders
                elif left_color is not None or right_color is not None:
                    if dist_to_left <= dist_to_right:
                        # Closer to left
                        if left_color is not None and j > 0:
                            result[i][j-1] = left_color
                    else:
                        # Closer to right
                        if right_color is not None and j < w - 1:
                            result[i][j+1] = right_color
    
    return result


def main():
    # Test on training examples
    task_path = '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/642248e4.json'
    
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    print(f"Testing solver on {len(task['train'])} training examples...")
    
    all_pass = True
    for idx, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        match = predicted_output == expected_output
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"Example {idx}: {status}")
        
        if not match:
            all_pass = False
            print(f"  Expected output shape: {len(expected_output)}x{len(expected_output[0])}")
            print(f"  Got output shape: {len(predicted_output)}x{len(predicted_output[0])}")
            
            # Show first few differences
            for i in range(len(expected_output)):
                for j in range(len(expected_output[0])):
                    if expected_output[i][j] != predicted_output[i][j]:
                        print(f"  Diff at ({i},{j}): expected {expected_output[i][j]}, got {predicted_output[i][j]}")
                        if i == 5:  # Show first 5 diffs
                            print("  ...")
                            break
                if i == 5:
                    break
    
    if all_pass:
        print("\n✓ All examples passed!")
        return 0
    else:
        print("\n✗ Some examples failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
