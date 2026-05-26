def solve(grid):
    """
    ARC-AGI Task 486c32d6: Pattern Anomaly Propagation
    
    Rule: When a row contains repeating pattern cells and some positions 
    within the pattern have anomalous (minority) values, propagate those 
    minority values to ALL corresponding positions across the row.
    """
    from collections import Counter
    
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Determine pattern length heuristically
    pattern_len = 4 if cols >= 20 else 3
    
    # Process each row
    for r in range(rows):
        row = grid[r]
        
        # Skip uniform rows (separators)
        if len(set(row)) <= 1:
            continue
        
        # Check if this row should be processed by looking for anomalies
        row_has_anomalies = False
        
        # First pass: detect if there are any anomalies in this row
        for pattern_pos in range(pattern_len):
            values_at_pos = []
            
            # Collect ALL values at this pattern position (including incomplete cycles)
            pos = pattern_pos
            while pos < cols:
                values_at_pos.append(row[pos])
                pos += pattern_len
            
            # If there are different values at this position, we have anomalies
            if len(set(values_at_pos)) > 1:
                row_has_anomalies = True
                break
        
        # Only process rows with anomalies
        if not row_has_anomalies:
            continue
        
        # Second pass: propagate anomalies
        for pattern_pos in range(pattern_len):
            values_at_pos = []
            indices_at_pos = []
            
            # Collect values and indices at this pattern position
            pos = pattern_pos
            while pos < cols:
                values_at_pos.append(row[pos])
                indices_at_pos.append(pos)
                pos += pattern_len
            
            if len(values_at_pos) < 2:
                continue
            
            # Count occurrences
            counter = Counter(values_at_pos)
            
            # If there are multiple values (anomalies)
            if len(counter) > 1:
                sorted_counts = counter.most_common()
                majority_val, majority_count = sorted_counts[0]
                
                # Find the minority value to propagate
                for minority_val, minority_count in sorted_counts[1:]:
                    if minority_count < majority_count:
                        # Propagate the minority value to ALL positions
                        for idx in indices_at_pos:
                            result[r][idx] = minority_val
                        break  # Only propagate the first minority value
    
    return result

# Full test
if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
        task_data = json.load(f)
    
    print("=== TESTING CORRECTED SOLVER ===")
    
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
            
            # Count different rows
            diff_rows = []
            for r in range(len(input_grid)):
                if actual_output[r] != expected_output[r]:
                    diff_rows.append(r)
            
            print(f"Different rows: {diff_rows}")
            
            # Show first few differences
            for r in diff_rows[:2]:
                print(f"  Row {r}:")
                print(f"    Input:    {input_grid[r]}")
                print(f"    Expected: {expected_output[r]}")
                print(f"    Actual:   {actual_output[r]}")
    
    print(f"\nOverall: {'✓ ALL PASS' if all_passed else '✗ NEED MORE DEBUG'}")