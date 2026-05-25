"""
ARC-AGI Task 505fff84 Solver

Rule: Extract the content between markers 1 and 8 (exclusive) on each row
that contains both markers. The markers are exclusive endpoints.
"""

import json
import sys


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Extract regions between 1 and 8 markers.
    
    For each row:
    - If it contains both a 1 and an 8, extract values between them (exclusive)
    - Otherwise, skip the row
    
    Args:
        grid: A 2D list of integers
        
    Returns:
        A 2D list where each row is the extracted content between markers
    """
    output = []
    for row in grid:
        pos_1 = None
        pos_8 = None
        
        for j, val in enumerate(row):
            if val == 1:
                pos_1 = j
            elif val == 8:
                pos_8 = j
        
        if pos_1 is not None and pos_8 is not None:
            start = min(pos_1, pos_8) + 1
            end = max(pos_1, pos_8)
            extracted = row[start:end]
            output.append(extracted)
    
    return output


if __name__ == "__main__":
    task_path = sys.argv[1] if len(sys.argv) > 1 else "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/505fff84.json"
    task_path = task_path.replace("~", "/Users/evanpieser")
    
    with open(task_path) as f:
        data = json.load(f)
    
    train_examples = data.get("train", [])
    
    print(f"Testing {len(train_examples)} training examples:")
    all_pass = True
    
    for idx, example in enumerate(train_examples):
        input_grid = example["input"]
        expected_output = example["output"]
        
        result = solve(input_grid)
        passed = result == expected_output
        all_pass = all_pass and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"  Example {idx}: {status}")
        
        if not passed:
            print(f"    Expected: {expected_output}")
            print(f"    Got:      {result}")
    
    if all_pass:
        print("\nAll training examples PASSED!")
    else:
        print("\nSome training examples FAILED!")
        sys.exit(1)
