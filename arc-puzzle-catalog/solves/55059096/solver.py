import json
import sys
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Find plus-shaped patterns and connect all 45-degree diagonal pairs with lines of 2s.
    """
    output = [row[:] for row in grid]
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    # Find all plus centers
    plus_centers = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if grid[r][c] == 3:
                if (grid[r-1][c] == 3 and grid[r+1][c] == 3 and 
                    grid[r][c-1] == 3 and grid[r][c+1] == 3):
                    plus_centers.append((r, c))
    
    plus_centers.sort()
    
    # Find all 45-degree diagonal pairs and draw them
    for i in range(len(plus_centers)):
        for j in range(i + 1, len(plus_centers)):
            r1, c1 = plus_centers[i]
            r2, c2 = plus_centers[j]
            
            dr = abs(r2 - r1)
            dc = abs(c2 - c1)
            
            # Only connect 45-degree diagonals
            if dr != dc:
                continue
            
            # Draw the diagonal line
            step_r = 1 if r2 > r1 else -1
            step_c = 1 if c2 > c1 else -1
            
            cr = r1 + step_r
            cc = c1 + step_c
            
            while (cr, cc) != (r2 - step_r, c2 - step_c):
                if 0 <= cr < height and 0 <= cc < width and output[cr][cc] == 0:
                    output[cr][cc] = 2
                cr += step_r
                cc += step_c
            
            # Add end point
            end_r = r2 - step_r
            end_c = c2 - step_c
            if 0 <= end_r < height and 0 <= end_c < width and output[end_r][end_c] == 0:
                output[end_r][end_c] = 2
    
    return output


def main():
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/55059096.json', 'r') as f:
        data = json.load(f)
    
    all_pass = True
    for idx, example in enumerate(data['train']):
        input_grid = example['input']
        expected_output = example['output']
        actual_output = solve(input_grid)
        
        if actual_output == expected_output:
            print(f"Training example {idx}: PASS")
        else:
            print(f"Training example {idx}: FAIL")
            all_pass = False
    
    if all_pass:
        print("\nAll training examples PASSED!")
    else:
        print("\nSome training examples FAILED!")
    
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
