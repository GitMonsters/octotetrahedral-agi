import json
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    ARC-AGI puzzle d492a647 solver.
    
    Pattern: Find the special marker (any value not 0 or 5).
    Replace all 0s that have the same row and column parity as the marker
    with the marker color.
    """
    # Create output grid as a copy of input
    result = [row[:] for row in grid]
    
    # Find the special marker (not 0 or 5)
    marker = None
    marker_r = None
    marker_c = None
    
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val not in [0, 5]:
                marker = val
                marker_r = r
                marker_c = c
                break
        if marker is not None:
            break
    
    # If no marker found, return grid as is
    if marker is None:
        return result
    
    # Get marker parities
    marker_row_parity = marker_r % 2
    marker_col_parity = marker_c % 2
    
    # Replace 0s with matching parity with the marker color
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == 0:
                row_parity = r % 2
                col_parity = c % 2
                
                if row_parity == marker_row_parity and col_parity == marker_col_parity:
                    result[r][c] = marker
    
    return result


if __name__ == '__main__':
    # Load and test on training examples
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/d492a647.json') as f:
        task = json.load(f)
    
    all_passed = True
    
    for idx, example in enumerate(task['train']):
        inp = example['input']
        expected = example['output']
        result = solve(inp)
        
        # Check if result matches expected
        passed = result == expected
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"Training example {idx + 1}: {status}")
        
        if not passed:
            # Find differences
            print("  Differences:")
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"    ({r}, {c}): got {result[r][c]}, expected {expected[r][c]}")
    
    print()
    if all_passed:
        print("All training examples passed! ✓")
    else:
        print("Some training examples failed.")
