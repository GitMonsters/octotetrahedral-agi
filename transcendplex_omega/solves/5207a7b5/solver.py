def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC-AGI Task 5207a7b5 Solver
    
    Pattern:
    1. Find the vertical line of 5's (column and length)
    2. Keep 5's in place
    3. Fill left of 5's with 8's in decreasing triangular pattern
    4. Fill right of 5's with 6's in decreasing triangular pattern
    """
    import copy
    result = copy.deepcopy(grid)
    
    height = len(grid)
    width = len(grid[0])
    
    # Find the vertical line of 5's
    col_5 = -1
    start_row = -1
    end_row = -1
    
    for col in range(width):
        for row in range(height):
            if grid[row][col] == 5:
                if col_5 == -1:
                    col_5 = col
                    start_row = row
                elif col_5 == col:
                    end_row = row
    
    if col_5 == -1:
        return result
    
    # Length of the 5's line
    line_length = end_row - start_row + 1
    
    # Fill left side with 8's
    # Pattern: full width for line_length + 2 rows, then decrease every 2 rows
    left_width = col_5
    
    for row_offset in range(height):
        row = start_row + row_offset
        if row >= height:
            break
        
        # Full width for first (line_length + 2) rows
        if row_offset < line_length + 2:
            current_left_width = left_width
        else:
            # After the extended line, decrease every 2 rows
            rows_after = row_offset - (line_length + 2)
            current_left_width = left_width - (rows_after + 2) // 2
        
        if current_left_width > 0:
            for j in range(current_left_width):
                result[row][j] = 8
    
    # Fill right side with 6's
    # Pattern: initial width = (line_length - 1) // 2
    # If line_length is ODD: decrease starts at row 1
    # If line_length is EVEN: decrease starts at row 2
    initial_right_width = (line_length - 1) // 2
    
    for row_offset in range(line_length):
        row = start_row + row_offset
        
        # Adjust offset for odd/even length
        if line_length % 2 == 1:  # ODD
            # Decrease at row 1: (row 0: 0, row 1: 1, row 2: 1, row 3: 2...)
            decreases = (row_offset + 1) // 2
        else:  # EVEN
            # Decrease at row 2: (row 0-1: 0, row 2-3: 1, row 4-5: 2...)
            decreases = row_offset // 2
        
        current_right_width = initial_right_width - decreases
        
        if current_right_width > 0 and col_5 + 1 + current_right_width <= width:
            for j in range(current_right_width):
                result[row][col_5 + 1 + j] = 6
    
    return result


if __name__ == "__main__":
    import json
    
    # Load test data
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5207a7b5.json") as f:
        data = json.load(f)
    
    # Test all training examples
    all_passed = True
    for idx, example in enumerate(data["train"]):
        input_grid = example["input"]
        expected_output = example["output"]
        predicted_output = solve(input_grid)
        
        if predicted_output == expected_output:
            print(f"✓ Training example {idx + 1} PASSED")
        else:
            print(f"✗ Training example {idx + 1} FAILED")
            print(f"\nExpected:")
            for row in expected_output:
                print(row)
            print(f"\nGot:")
            for row in predicted_output:
                print(row)
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TRAINING EXAMPLES PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
