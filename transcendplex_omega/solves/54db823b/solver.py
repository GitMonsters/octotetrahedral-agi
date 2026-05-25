"""
Solver for ARC-AGI task 54db823b

Rule: Find all connected components (blocks) that contain both 3 and 9.
Among these mixed blocks, identify the one with the lowest count of 9s.
Remove (set to 0) all cells in that block.
"""

import json
from scipy import ndimage
import numpy as np


def solve(grid):
    """
    Solve ARC task 54db823b.
    
    Args:
        grid: List of lists representing the input grid
        
    Returns:
        List of lists representing the output grid
    """
    arr = np.array(grid, dtype=int)
    nonzero = arr > 0
    labeled, num_features = ndimage.label(nonzero)
    
    # Find all blocks with both 3 and 9, and their 9-counts
    mixed_blocks = []
    
    for label in range(1, num_features + 1):
        coords = np.where(labeled == label)
        if len(coords[0]) > 0:
            vals = arr[labeled == label]
            
            if 3 in vals and 9 in vals:
                count_9 = int((vals == 9).sum())
                mixed_blocks.append((label, count_9))
    
    # Clear the block with the lowest count of 9s
    result = arr.copy()
    if mixed_blocks:
        target_label = min(mixed_blocks, key=lambda x: x[1])[0]
        result[labeled == target_label] = 0
    
    return result.tolist()


if __name__ == '__main__':
    # Load task JSON
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/54db823b.json') as f:
        data = json.load(f)
    
    # Test on all training examples
    print("Testing solver on training examples:")
    all_pass = True
    
    for ex_idx in range(len(data['train'])):
        example = data['train'][ex_idx]
        inp = example['input']
        expected_out = example['output']
        
        predicted_out = solve(inp)
        
        # Check if prediction matches expected output
        match = all(predicted_out[i][j] == expected_out[i][j] 
                    for i in range(len(predicted_out)) 
                    for j in range(len(predicted_out[0])))
        
        status = 'PASS' if match else 'FAIL'
        print(f"Training example {ex_idx}: {status}")
        
        if not match:
            all_pass = False
    
    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
