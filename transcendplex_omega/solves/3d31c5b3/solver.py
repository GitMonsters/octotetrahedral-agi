"""
ARC-AGI Task 3d31c5b3 Solver

Pattern: Overlay four layers (5, 4, 2, 8) with priority 5 > 4 > 8 > 2
- Input: 12x6 grid (4 stacked 3x6 layers)
- Output: 3x6 grid (composite result)

For each position, pick the non-zero value with highest priority.
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve ARC-AGI task 3d31c5b3.
    
    Args:
        grid: 12x6 input grid with 4 stacked layers
    
    Returns:
        3x6 output grid
    """
    # Extract the 4 layers from the input
    # Layer 5: rows 0-2
    # Layer 4: rows 3-5
    # Layer 2: rows 6-8
    # Layer 8: rows 9-11
    
    result = []
    
    for i in range(3):
        row = []
        for j in range(6):
            l5 = grid[i][j]
            l4 = grid[3 + i][j]
            l2 = grid[6 + i][j]
            l8 = grid[9 + i][j]
            
            # Priority: 5 > 4 > 8 > 2
            if l5 != 0:
                value = l5
            elif l4 != 0:
                value = l4
            elif l8 != 0:
                value = l8
            elif l2 != 0:
                value = l2
            else:
                value = 0
            
            row.append(value)
        
        result.append(row)
    
    return result


if __name__ == "__main__":
    import json
    import sys
    
    # Load task JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/3d31c5b3.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on training examples
    print("Testing on training examples:")
    all_pass = True
    
    for i, example in enumerate(task['train']):
        predicted = solve(example['input'])
        expected = example['output']
        
        match = predicted == expected
        status = "PASS" if match else "FAIL"
        print(f"  Example {i}: {status}")
        
        if not match:
            all_pass = False
            print(f"    Expected: {expected}")
            print(f"    Got:      {predicted}")
    
    print()
    if all_pass:
        print("✓ ALL TRAINING EXAMPLES PASSED!")
    else:
        print("✗ Some training examples failed")
        sys.exit(1)
