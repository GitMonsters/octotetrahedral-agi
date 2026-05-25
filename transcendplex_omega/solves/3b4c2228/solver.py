#!/usr/bin/env python3
"""
Solver for ARC-AGI task 3b4c2228.

Rule: Count 2x2 blocks of value 3 in the input grid.
Create a 3x3 output grid with 1s on the main diagonal,
with the number of 1s equal to the count of 2x2 blocks of 3.
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solve ARC task 3b4c2228.
    
    Args:
        grid: Input grid as list of lists of integers
        
    Returns:
        3x3 output grid with 1s on diagonal
    """
    # Count 2x2 blocks of value 3
    block_count = 0
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    
    for r in range(h - 1):
        for c in range(w - 1):
            if (grid[r][c] == 3 and grid[r][c + 1] == 3 and
                grid[r + 1][c] == 3 and grid[r + 1][c + 1] == 3):
                block_count += 1
    
    # Create 3x3 output with 1s on diagonal based on block_count
    output = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(min(block_count, 3)):
        output[i][i] = 1
    
    return output


if __name__ == "__main__":
    import json
    import sys
    
    # Load task JSON
    task_path = sys.argv[1] if len(sys.argv) > 1 else "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/3b4c2228.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path) as f:
        task = json.load(f)
    
    # Test on training examples
    print("Testing on training examples:")
    print("=" * 60)
    
    all_pass = True
    for idx, example in enumerate(task['train']):
        inp = example['input']
        expected = example['output']
        result = solve(inp)
        
        passed = result == expected
        all_pass = all_pass and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"Training example {idx + 1}: {status}")
        if not passed:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
    
    # Test on test examples if available
    if 'test' in task and task['test']:
        print("\nTesting on test examples:")
        print("=" * 60)
        for idx, example in enumerate(task['test']):
            inp = example['input']
            result = solve(inp)
            print(f"Test example {idx + 1}: {result}")
    
    print("\n" + "=" * 60)
    if all_pass:
        print("All training examples PASSED ✓")
    else:
        print("Some training examples FAILED ✗")
        sys.exit(1)
