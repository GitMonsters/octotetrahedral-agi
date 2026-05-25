import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solves ARC-AGI task 5d2a5c43.
    
    The input grid is divided into left and right sections by a column of 1s.
    The rule is to perform a logical OR operation on the two sections:
    - If either left[i][j] == 4 OR right[i][j] == 4, output 8
    - Otherwise, output 0
    """
    # Find the divider column (column of 1s)
    divider_col = None
    for col in range(len(grid[0])):
        if all(grid[row][col] == 1 for row in range(len(grid))):
            divider_col = col
            break
    
    if divider_col is None:
        raise ValueError("No divider column found")
    
    # Split into left and right sections
    left = [row[:divider_col] for row in grid]
    right = [row[divider_col + 1:] for row in grid]
    
    # Apply OR operation: if either side has 4, output 8; else 0
    result = []
    for row_idx in range(len(left)):
        new_row = []
        for col_idx in range(len(left[row_idx])):
            left_val = left[row_idx][col_idx]
            right_val = right[row_idx][col_idx]
            
            if left_val == 4 or right_val == 4:
                new_row.append(8)
            else:
                new_row.append(0)
        result.append(new_row)
    
    return result


if __name__ == "__main__":
    # Load the task JSON
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5d2a5c43.json") as f:
        task = json.load(f)
    
    # Test all training examples
    print("Testing training examples:")
    all_pass = True
    for idx, example in enumerate(task["train"]):
        result = solve(example["input"])
        expected = example["output"]
        passed = result == expected
        all_pass = all_pass and passed
        print(f"  Train {idx}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"    Expected: {expected}")
            print(f"    Got: {result}")
    
    # Test all test examples
    print("Testing test examples:")
    for idx, example in enumerate(task["test"]):
        result = solve(example["input"])
        expected = example["output"]
        passed = result == expected
        all_pass = all_pass and passed
        print(f"  Test {idx}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"    Expected: {expected}")
            print(f"    Got: {result}")
    
    print()
    print("=" * 50)
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 50)
