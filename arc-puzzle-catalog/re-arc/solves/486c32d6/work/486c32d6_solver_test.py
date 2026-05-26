def solve(grid):
    """
    ARC-AGI Task 486c32d6: Pattern Propagation
    
    Rule: In grids with separator columns, propagate anomalous values
    to their corresponding positions across all pattern cycles in each row.
    """
    from collections import Counter
    
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Find separator columns (should be uniform color)
    separator_cols = []
    for c in range(cols):
        col_values = [grid[r][c] for r in range(rows)]
        if len(set(col_values)) <= 2:  # Almost uniform (allowing minor variations)
            separator_cols.append(c)
    
    # Determine pattern length by finding repeating separator positions
    pattern_len = None
    if len(separator_cols) >= 2:
        # Try to find regular spacing
        spacings = []
        for i in range(1, len(separator_cols)):
            spacings.append(separator_cols[i] - separator_cols[i-1])
        
        # Use most common spacing as pattern length
        if spacings:
            counter = Counter(spacings)
            pattern_len = counter.most_common(1)[0][0]
    
    # Fallback: try common pattern lengths
    if pattern_len is None:
        for test_len in [3, 4, 5]:
            # Check if this creates regular separators
            expected_seps = list(range(test_len-1, cols, test_len))
            if len(expected_seps) >= 2:
                # Check if these positions are actually separators
                is_pattern = True
                for sep_pos in expected_seps[:3]:  # Check first few
                    if sep_pos >= cols:
                        continue
                    col_vals = [grid[r][sep_pos] for r in range(rows)]
                    if len(set(col_vals)) > 2:  # Not uniform enough
                        is_pattern = False
                        break
                if is_pattern:
                    pattern_len = test_len
                    break
    
    if pattern_len is None:
        return result  # No pattern found
    
    # Process each row
    for r in range(rows):
        row = grid[r]
        
        # Skip separator rows (horizontal separators)
        if len(set(row)) <= 2:
            continue
        
        # Find complete pattern cycles
        complete_cycles = cols // pattern_len
        if complete_cycles < 2:
            continue
            
        # For each position in the pattern
        for pattern_pos in range(pattern_len):
            # Collect values at this position across all cycles
            values = []
            positions = []
            for cycle in range(complete_cycles):
                pos = cycle * pattern_len + pattern_pos
                if pos < cols:
                    values.append(row[pos])
                    positions.append(pos)
            
            if not values:
                continue
                
            # Count occurrences
            counter = Counter(values)
            
            # If there are anomalies (multiple different values)
            if len(counter) > 1:
                sorted_counts = counter.most_common()
                majority_val = sorted_counts[0][0]
                majority_count = sorted_counts[0][1]
                
                # For each minority value
                for val, count in sorted_counts[1:]:
                    if count < majority_count:
                        # Propagate this minority value to ALL positions in this pattern
                        for pos in positions:
                            result[r][pos] = val
                        break  # Only propagate the first (most common) minority value
    
    return result

# Test with training examples
if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
        task_data = json.load(f)
    
    print("Testing solver on training examples:")
    
    for i, example in enumerate(task_data['train']):
        print(f"\nTrain {i}:")
        input_grid = example['input']
        expected_output = example['output']
        actual_output = solve(input_grid)
        
        # Check if correct
        is_correct = actual_output == expected_output
        print(f"Result: {'✓ PASS' if is_correct else '✗ FAIL'}")
        
        if not is_correct:
            print("First few differences:")
            for r in range(min(5, len(input_grid))):
                if actual_output[r] != expected_output[r]:
                    print(f"  Row {r}:")
                    print(f"    Expected: {expected_output[r]}")
                    print(f"    Got:      {actual_output[r]}")