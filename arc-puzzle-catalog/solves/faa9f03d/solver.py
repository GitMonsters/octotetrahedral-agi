import json
import sys
from copy import deepcopy

def solve(grid):
    """
    Simple pattern observed from examples:
    1. Replace 2s based on neighbors
    2. Extend lines to fill gaps between same colors
    """
    result = deepcopy(grid)
    rows, cols = len(grid), len(grid[0])
    
    # Step 1: Replace all 2s
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                # Look at 4-directional neighbors for non-zero values
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        val = grid[nr][nc]
                        if val != 0 and val != 2:
                            neighbors.append(val)
                
                if neighbors:
                    # Use the most frequent non-2 neighbor
                    from collections import Counter
                    result[r][c] = Counter(neighbors).most_common(1)[0][0]
                else:
                    result[r][c] = 0
    
    # Step 2: Fill gaps to extend lines - be very conservative
    # Only fill if there are exactly matching colors on opposite sides
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 0:
                # Check horizontal extension
                left = result[r][c-1] if c > 0 else None
                right = result[r][c+1] if c < cols-1 else None
                
                if left is not None and left == right and left != 0:
                    result[r][c] = left
                    continue
                
                # Check vertical extension
                up = result[r-1][c] if r > 0 else None
                down = result[r+1][c] if r < rows-1 else None
                
                if up is not None and up == down and up != 0:
                    result[r][c] = up
    
    # Step 3: Handle special case where non-2 values need to be overridden
    # Look at example 1: position (3,3) changes from 3 to 1
    # This happens when a line needs to pass through
    for r in range(rows):
        for c in range(cols):
            current_val = result[r][c]
            if current_val != 0 and grid[r][c] != 2:  # Non-zero, non-2 original value
                
                # Check if there's a horizontal line that should pass through here
                left_context = []
                right_context = []
                
                # Look left for a pattern
                for cc in range(c-1, max(-1, c-4), -1):  # Look 3 positions left
                    if result[r][cc] != 0 and result[r][cc] != current_val:
                        left_context.append(result[r][cc])
                        break
                
                # Look right for a pattern  
                for cc in range(c+1, min(cols, c+4)):  # Look 3 positions right
                    if result[r][cc] != 0 and result[r][cc] != current_val:
                        right_context.append(result[r][cc])
                        break
                
                # If same non-current color on both sides, this might be a line extension
                if (len(left_context) == 1 and len(right_context) == 1 and
                    left_context[0] == right_context[0]):
                    # Check if there are 2s or 0s in between that would justify this
                    has_gap = False
                    for cc in range(max(0, c-3), min(cols, c+3)):
                        if grid[r][cc] == 2 or grid[r][cc] == 0:
                            has_gap = True
                            break
                    
                    if has_gap:
                        result[r][c] = left_context[0]
    
    # Step 4: One more round of gap filling
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 0:
                left = result[r][c-1] if c > 0 else None
                right = result[r][c+1] if c < cols-1 else None
                
                if left is not None and left == right and left != 0:
                    result[r][c] = left
                    continue
                    
                up = result[r-1][c] if r > 0 else None
                down = result[r+1][c] if r < rows-1 else None
                
                if up is not None and up == down and up != 0:
                    result[r][c] = up
    
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python solver.py <path_to_json>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    # Test on training examples
    for i, example in enumerate(data['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        if predicted_output == expected_output:
            print(f"Training example {i+1}: PASS")
        else:
            print(f"Training example {i+1}: FAIL")
            print("Expected:")
            for row in expected_output:
                print(row)
            print("Got:")
            for row in predicted_output:
                print(row)
            print()
    
    # Test on test examples if present
    if 'test' in data:
        for i, example in enumerate(data['test']):
            input_grid = example['input']
            expected_output = example['output']
            predicted_output = solve(input_grid)
            
            if predicted_output == expected_output:
                print(f"Test example {i+1}: PASS")
            else:
                print(f"Test example {i+1}: FAIL")
                print("Expected:")
                for row in expected_output:
                    print(row)
                print("Got:")
                for row in predicted_output:
                    print(row)
                print()