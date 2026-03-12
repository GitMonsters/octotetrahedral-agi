#!/usr/bin/env python3
"""
ARC-AGI Task 47996f11 Solver

The task involves replacing cells with value 6 with appropriate values.
The rule appears to be based on weighted voting from non-6 neighboring cells,
where closer cells have higher weight.
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Replace all 6 values with values determined by weighted neighbor voting.
    
    For each cell with value 6:
    - Look at all non-6 neighbors within a radius of 4
    - Weight each neighbor by the inverse square of its distance (Chebyshev distance)
    - Replace the 6 with the neighbor value that has the highest total weight
    """
    h = len(grid)
    w = len(grid[0])
    
    # Make a copy to avoid modifying the input
    output = [row[:] for row in grid]
    
    # Process each cell
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 6:
                # Calculate weighted vote from neighbors
                weighted_vals = {}
                
                # Look at neighbors within radius 4
                for dr in range(-4, 5):
                    for dc in range(-4, 5):
                        if dr == 0 and dc == 0:
                            continue
                        
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            val = grid[nr][nc]
                            if val != 6:
                                # Weight inversely by Chebyshev distance squared
                                dist = max(abs(dr), abs(dc))
                                weight = 1.0 / (dist * dist)
                                weighted_vals[val] = weighted_vals.get(val, 0) + weight
                
                # Fill with the value that has highest total weight
                if weighted_vals:
                    best_val = max(weighted_vals, key=weighted_vals.get)
                    output[r][c] = best_val
    
    return output


def main():
    import json
    import sys
    
    # Load task JSON
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = os.path.expanduser('~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/47996f11.json')
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for train_idx, example in enumerate(data['train']):
        inp = example['input']
        expected = example['output']
        
        predicted = solve(inp)
        
        # Check if outputs match
        if predicted == expected:
            print(f"Training example {train_idx}: PASS")
        else:
            # Count mismatches
            mismatches = 0
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if predicted[r][c] != expected[r][c]:
                        mismatches += 1
            
            print(f"Training example {train_idx}: FAIL ({mismatches} mismatches)")
            all_pass = False
    
    if all_pass:
        print("\nAll training examples PASSED")
    else:
        print("\nSome training examples FAILED")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    import os
    exit(main())
