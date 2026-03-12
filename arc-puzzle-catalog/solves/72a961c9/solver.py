def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: Find a row containing mostly 1s with some non-1 values.
    For each non-1 value at column position, draw a vertical line:
    - The non-1 color appears at a height based on column position and neighboring non-1s
    - 1s fill the space between the non-1 color and the base row
    """
    result = [row[:] for row in grid]
    
    # Find the row that acts as the "base" (contains mostly 1s with some other colors)
    base_row = None
    base_row_idx = None
    
    for i, row in enumerate(grid):
        ones_count = sum(1 for cell in row if cell == 1)
        # If most of the row is 1s and there are other colors
        if ones_count >= len(row) - 3 and ones_count < len(row):
            base_row = row
            base_row_idx = i
            break
    
    if base_row is None:
        return result
    
    # Find all non-1 positions and their values
    non1_cols = [col for col in range(len(base_row)) if base_row[col] != 1]
    
    if not non1_cols:
        return result
    
    # For each non-1 position, calculate how far up the line should extend
    width = len(base_row)
    for idx, col in enumerate(non1_cols):
        color = base_row[col]
        
        # Count non-1s to left and right
        non1_left = sum(1 for j in range(col) if base_row[j] != 1)
        non1_right = sum(1 for j in range(col + 1, width) if base_row[j] != 1)
        
        # Distance from edges
        dist_from_left = col
        dist_from_right = width - col - 1
        min_dist_from_edge = min(dist_from_left, dist_from_right)
        
        # Determine position in sequence
        is_last = (idx == len(non1_cols) - 1)
        is_only = (len(non1_cols) == 1)
        
        # Determine how far up to draw the line
        if is_only:
            # Single non-1 in entire row
            if dist_from_left == dist_from_right:
                # Centered
                distance_up = min_dist_from_edge + 1
            else:
                # Off-center: use all rows above
                distance_up = base_row_idx
        elif non1_left == 0:
            # Non-1s only to the right
            # Count continuous 1s to the right until next non-1
            ones_to_right = 0
            for j in range(col + 1, width):
                if base_row[j] != 1:
                    break
                ones_to_right += 1
            distance_up = ones_to_right + 1
        elif non1_right == 0:
            # Non-1s only to the left
            if is_last:
                # Last position with non-1s only to left
                # Count continuous 1s to the left
                ones_to_left = 0
                for j in range(col - 1, -1, -1):
                    if base_row[j] != 1:
                        break
                    ones_to_left += 1
                
                # Use base - non1_left if ones_to_left is very small, otherwise ones_to_left
                if ones_to_left <= 1:
                    distance_up = base_row_idx - non1_left
                else:
                    distance_up = ones_to_left
            else:
                # Shouldn't happen
                distance_up = dist_from_right
        else:
            # Non-1s on both sides
            distance_up = min_dist_from_edge + 1
        
        # Draw the line: non-1 color at top, 1s filling down
        top_row = max(0, base_row_idx - distance_up)
        result[top_row][col] = color
        for row_idx in range(top_row + 1, base_row_idx):
            result[row_idx][col] = 1
    
    return result


if __name__ == "__main__":
    # Training examples
    examples = [
        {
            "input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 2, 1, 1, 1, 8, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            "expected": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 8, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 2, 1, 1, 1, 8, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        },
        {
            "input": [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 2, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
            "expected": [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 2, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
        },
        {
            "input": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 8, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            "expected": [[0, 8, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 8, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        },
        {
            "input": [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 8, 1, 1, 1, 8, 1, 2, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            "expected": [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0], [0, 8, 0, 0, 0, 8, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0], [1, 8, 1, 1, 1, 8, 1, 2, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, example in enumerate(examples):
        result = solve(example["input"])
        expected = example["expected"]
        
        if result == expected:
            print(f"✓ Training example {i + 1} PASSED")
            passed += 1
        else:
            print(f"✗ Training example {i + 1} FAILED")
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    if failed == 0:
        print("All training examples passed!")
