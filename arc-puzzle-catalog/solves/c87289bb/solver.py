def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: Extend 8-columns downward, except those inside the 2-region.
    For the 2-region, create a box by:
    1. Top border from min_2_col to the nearest 8 on the right
    2. Side walls extending downward from adjacent columns to nearest outer 8s
    """
    result = [row[:] for row in grid]
    
    # Find 2-region bounds
    rows_with_2 = []
    cols_with_2 = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                rows_with_2.append(i)
                cols_with_2.append(j)
    
    # Find all 8-columns and last row with 8
    cols_with_8 = set()
    last_row_with_8 = -1
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 8:
                cols_with_8.add(j)
                last_row_with_8 = max(last_row_with_8, i)
    
    if not cols_with_8:
        return result
    
    if not rows_with_2:
        # No 2s, just extend all 8-columns
        for col in cols_with_8:
            for row in range(last_row_with_8 + 1, len(grid)):
                if result[row][col] == 0:
                    result[row][col] = 8
        return result
    
    min_2_row = min(rows_with_2)
    max_2_row = max(rows_with_2)
    min_2_col = min(cols_with_2)
    max_2_col = max(cols_with_2)
    
    # Extend 8-columns that are OUTSIDE the 2-region
    cols_2_set = set(cols_with_2)
    cols_8_outside_2 = cols_with_8 - cols_2_set
    for col in cols_8_outside_2:
        for row in range(last_row_with_8 + 1, len(grid)):
            if result[row][col] == 0:
                result[row][col] = 8
    
    # Create frame around 2-region
    top_frame_row = min_2_row - 1
    if top_frame_row >= 0:
        # Top border: from the 2-region to right_8-1
        right_8 = min([c for c in cols_with_8 if c > max_2_col], default=max(cols_with_8))
        
        # Determine frame left edge:
        # If min_2_col has an 8 in input, include min_2_col-1; otherwise start at min_2_col
        frame_left = min_2_col - 1 if min_2_col in cols_with_8 else min_2_col
        frame_left = max(0, frame_left)
        
        # Fill cols from frame_left to right_8-1, excluding cols immediately before 8s
        for col in range(frame_left, right_8):
            # Skip if immediately before an 8-col
            if col + 1 not in cols_with_8:
                if result[top_frame_row][col] == 0:
                    result[top_frame_row][col] = 8
    
    # Extend columns from the frame top downward
    # Find which columns have 8 in the top frame row (from the frame)
    cols_in_frame_top = set()
    if top_frame_row >= 0:
        cols_in_frame_top = set(j for j in range(len(grid[0])) if result[top_frame_row][j] == 8)
    
    # Extend columns that are in the frame top (but not already 8-cols)
    frame_only_cols = cols_in_frame_top - cols_with_8
    for col in frame_only_cols:
        for row in range(min_2_row, len(grid)):
            if result[row][col] == 0:
                result[row][col] = 8
    
    return result


if __name__ == "__main__":
    import json
    import os
    
    # Load test data
    with open(os.path.expanduser('~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/c87289bb.json')) as f:
        data = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for idx, example in enumerate(data['train']):
        result = solve(example['input'])
        expected = example['output']
        
        if result == expected:
            print(f"✓ Example {idx+1} PASSED")
        else:
            print(f"✗ Example {idx+1} FAILED")
            all_pass = False
            # Show differences
            for i in range(len(result)):
                for j in range(len(result[0])):
                    if result[i][j] != expected[i][j]:
                        print(f"  ({i},{j}): got {result[i][j]}, expected {expected[i][j]}")
    
    if all_pass:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
