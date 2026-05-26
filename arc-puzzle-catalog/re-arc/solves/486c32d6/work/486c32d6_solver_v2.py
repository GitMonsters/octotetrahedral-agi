def solve(grid):
    """
    ARC-AGI Task 486c32d6: Pattern Propagation
    
    Rule: In grids with repeating pattern cells, when a row has anomalous values
    at certain positions within pattern cells, propagate those anomalous values
    to the same positions across all pattern cells in that row.
    """
    from collections import Counter
    
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Detect pattern length by analyzing the first non-uniform row
    pattern_len = None
    for r in range(rows):
        row = grid[r]
        if len(set(row)) > 1:  # Not uniform
            # Try different pattern lengths
            for test_len in [3, 4, 5]:
                if test_len * 2 > cols:
                    continue
                    
                # Check if row follows this pattern
                cycles = cols // test_len
                if cycles < 2:
                    continue
                
                # Check consistency across cycles
                is_valid_pattern = True
                for pos in range(test_len):
                    values_at_pos = []
                    for cycle in range(cycles):
                        idx = cycle * test_len + pos
                        if idx < cols:
                            values_at_pos.append(row[idx])
                    
                    # For a valid pattern, most positions should have the same value
                    counter = Counter(values_at_pos)
                    if len(counter) > 1:
                        # Check if there's a clear majority (pattern with anomalies)
                        most_common = counter.most_common()
                        if most_common[0][1] < len(values_at_pos) * 0.6:  # Less than 60% majority
                            is_valid_pattern = False
                            break
                
                if is_valid_pattern:
                    pattern_len = test_len
                    break
            
            if pattern_len:
                break
    
    if pattern_len is None:
        return result  # No pattern detected
    
    # Process each row
    for r in range(rows):
        row = grid[r]
        
        # Skip uniform rows (likely separators)
        if len(set(row)) <= 1:
            continue
        
        cycles = cols // pattern_len
        if cycles < 2:
            continue
        
        # Check each position within the pattern
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
            
            # Count occurrences
            counter = Counter(values_at_pos)
            
            # If there are anomalies (not all the same)
            if len(counter) > 1:
                sorted_counts = counter.most_common()
                majority_val, majority_count = sorted_counts[0]
                
                # Find minority values to propagate
                for minority_val, minority_count in sorted_counts[1:]:
                    if minority_count < majority_count:
                        # Propagate minority value to ALL positions in this pattern
                        for idx in indices_at_pos:
                            result[r][idx] = minority_val
                        break  # Only propagate the most frequent minority value
    
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
            print("Differences found:")
            diff_count = 0
            for r in range(len(input_grid)):
                if actual_output[r] != expected_output[r]:
                    diff_count += 1
                    if diff_count <= 3:  # Show first 3 different rows
                        print(f"  Row {r}:")
                        print(f"    Input:    {input_grid[r]}")
                        print(f"    Expected: {expected_output[r]}")
                        print(f"    Got:      {actual_output[r]}")
    
    print(f"\nOverall result: {'✓ ALL PASS' if all_passed else '✗ SOME FAILED'}")