def solve(grid):
    """
    ARC-AGI Task 486c32d6: Pattern Anomaly Propagation
    
    Rule: When a row has repeating patterns and there are anomalous values
    at certain positions within the pattern, propagate those anomalous values
    to the same positions in ALL pattern cycles.
    """
    from collections import Counter
    
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Detect pattern length by analyzing the structure
    # Look for regular separators or repeating structures
    pattern_len = None
    
    # Try to find pattern by looking at first row with variations
    for r in range(rows):
        row = grid[r]
        if len(set(row)) > 1:  # Not uniform
            # Try different pattern lengths
            for test_len in [3, 4, 5]:
                if test_len >= cols:
                    continue
                    
                cycles = cols // test_len
                if cycles < 2:
                    continue
                
                # Check if this creates a valid repeating pattern
                is_pattern = True
                for pos in range(test_len):
                    values_at_pos = []
                    for cycle in range(cycles):
                        idx = cycle * test_len + pos
                        if idx < cols:
                            values_at_pos.append(row[idx])
                    
                    # For most positions, values should be mostly the same
                    counter = Counter(values_at_pos)
                    if len(counter) > 2:  # Too many different values
                        is_pattern = False
                        break
                
                if is_pattern:
                    pattern_len = test_len
                    break
            
            if pattern_len:
                break
    
    if pattern_len is None:
        return result  # No pattern found
    
    # Process each row
    for r in range(rows):
        row = grid[r]
        
        # Skip uniform rows
        if len(set(row)) <= 1:
            continue
        
        cycles = cols // pattern_len
        if cycles < 2:
            continue
        
        # For each position in the pattern
        for pattern_pos in range(pattern_len):
            values_at_pos = []
            indices_at_pos = []
            
            for cycle in range(cycles):
                idx = cycle * pattern_len + pattern_pos
                if idx < cols:
                    values_at_pos.append(row[idx])
                    indices_at_pos.append(idx)
            
            if not values_at_pos:
                continue
            
            # Count occurrences at this pattern position
            counter = Counter(values_at_pos)
            
            # If there are multiple values (anomalies)
            if len(counter) > 1:
                # Find the minority value(s) that should be propagated
                sorted_counts = counter.most_common()
                majority_val, majority_count = sorted_counts[0]
                
                # Look for minority values to propagate
                for minority_val, minority_count in sorted_counts[1:]:
                    if minority_count > 0 and minority_count < majority_count:
                        # Propagate this minority value to ALL positions
                        for idx in indices_at_pos:
                            result[r][idx] = minority_val
                        break  # Only propagate the first (most common) minority
    
    return result

# Test with training examples
if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
        task_data = json.load(f)
    
    print("Testing solver on training examples:")
    
    all_passed = True
    for i, example in enumerate(task_data['train']):
        print(f"\nTrain {i}:")
        input_grid = example['input']
        expected_output = example['output']
        actual_output = solve(input_grid)
        
        # Check if correct
        is_correct = actual_output == expected_output
        print(f"Result: {'✓ PASS' if is_correct else '✗ FAIL'}")
        
        if not is_correct:
            all_passed = False
            # Show specific differences
            changed_rows = []
            for r in range(len(input_grid)):
                if actual_output[r] != expected_output[r]:
                    changed_rows.append(r)
            
            print(f"Different rows: {changed_rows[:5]}")
            
            # Show first differing row in detail
            if changed_rows:
                r = changed_rows[0]
                print(f"Row {r} details:")
                print(f"  Input:    {input_grid[r]}")
                print(f"  Expected: {expected_output[r]}")
                print(f"  Got:      {actual_output[r]}")
    
    print(f"\nOverall result: {'✓ ALL PASS' if all_passed else '✗ SOME FAILED'}")