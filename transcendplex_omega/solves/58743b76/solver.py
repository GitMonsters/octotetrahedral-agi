import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Puzzle 58743b76: Replace values in main grid based on quadrant mapping.
    The metadata 2x2 defines the replacement values for each quadrant.
    """
    grid = [row[:] for row in grid]  # Deep copy
    
    rows, cols = len(grid), len(grid[0])
    
    # Find metadata: look for a 2x2 area with non-0, non-8 values
    metadata = None
    metadata_pos = None
    
    # Check all four corners (2x2 regions)
    corners = [
        ((0, 2), (0, 2), "top-left"),
        ((0, 2), (cols-2, cols), "top-right"),
        ((rows-2, rows), (0, 2), "bottom-left"),
        ((rows-2, rows), (cols-2, cols), "bottom-right"),
    ]
    
    for row_range, col_range, corner_name in corners:
        candidate = [
            [grid[row_range[0]][col_range[0]], grid[row_range[0]][col_range[1]-1]],
            [grid[row_range[1]-1][col_range[0]], grid[row_range[1]-1][col_range[1]-1]]
        ]
        # Check if this looks like metadata (non-8 values)
        if all(candidate[i][j] != 8 for i in range(2) for j in range(2)):
            metadata = candidate
            metadata_pos = corner_name
            break
    
    if metadata is None:
        return grid
    
    # Based on metadata position, determine main grid bounds
    if metadata_pos == "top-left":
        main_rows = (2, rows)
        main_cols = (2, cols)
    elif metadata_pos == "top-right":
        main_rows = (2, rows)
        main_cols = (0, cols - 2)
    elif metadata_pos == "bottom-left":
        main_rows = (0, rows - 2)
        main_cols = (2, cols)
    else:  # bottom-right
        main_rows = (0, rows - 2)
        main_cols = (0, cols - 2)
    
    # Extract metadata values for each quadrant
    tl_val = metadata[0][0]  # top-left quadrant value
    tr_val = metadata[0][1]  # top-right quadrant value
    bl_val = metadata[1][0]  # bottom-left quadrant value
    br_val = metadata[1][1]  # bottom-right quadrant value
    
    # Calculate midpoint of main grid
    mid_row = main_rows[0] + (main_rows[1] - main_rows[0]) // 2
    mid_col = main_cols[0] + (main_cols[1] - main_cols[0]) // 2
    
    # Replace ALL non-0, non-8 values based on quadrant
    for i in range(main_rows[0], main_rows[1]):
        for j in range(main_cols[0], main_cols[1]):
            val = grid[i][j]
            if val not in (0, 8):  # Replace all non-0, non-8 values
                if i < mid_row and j < mid_col:
                    grid[i][j] = tl_val
                elif i < mid_row and j >= mid_col:
                    grid[i][j] = tr_val
                elif i >= mid_row and j < mid_col:
                    grid[i][j] = bl_val
                else:  # bottom-right
                    grid[i][j] = br_val
    
    return grid


if __name__ == "__main__":
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/58743b76.json', 'r') as f:
        task = json.load(f)
    
    print("Testing training examples:")
    all_pass = True
    for idx, example in enumerate(task['train']):
        result = solve(example['input'])
        expected = example['output']
        
        passed = result == expected
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  Training {idx + 1}: {status}")
        
        if not passed:
            # Print first few differences
            count = 0
            for i in range(len(result)):
                for j in range(len(result[0])):
                    if result[i][j] != expected[i][j]:
                        print(f"    Diff at ({i}, {j}): got {result[i][j]}, expected {expected[i][j]}")
                        count += 1
                        if count >= 3:
                            break
                if count >= 3:
                    break
    
    if all_pass:
        print("\n✓ All training examples passed!")
    else:
        print("\n✗ Some training examples failed")
