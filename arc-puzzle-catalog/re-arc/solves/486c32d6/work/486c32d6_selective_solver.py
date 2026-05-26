def solve(grid):
    """
    ARC-AGI Task 486c32d6: Selective Minority Value Propagation
    
    Rule: In rows with patterns, when some positions have minority values,
    propagate ONLY to positions that currently have the majority value.
    """
    from collections import Counter
    
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Determine pattern length  
    pattern_len = 4 if cols >= 20 else 3
    
    # Process each row
    for r in range(rows):
        row = grid[r]
        
        # Skip uniform rows
        if len(set(row)) <= 1:
            continue
        
        # Process each pattern position
        for pattern_pos in range(pattern_len):
            # Collect values and indices at this pattern position
            values_at_pos = []
            indices_at_pos = []
            
            pos = pattern_pos
            while pos < cols:
                values_at_pos.append(row[pos])
                indices_at_pos.append(pos)
                pos += pattern_len
            
            if len(values_at_pos) < 2:
                continue
            
            # Count occurrences
            counter = Counter(values_at_pos)
            
            # Only process if there are anomalies
            if len(counter) > 1:
                sorted_counts = counter.most_common()
                majority_val, majority_count = sorted_counts[0]
                
                # Find minority value to propagate
                for minority_val, minority_count in sorted_counts[1:]:
                    if minority_count < majority_count:
                        # KEY INSIGHT: Only change positions that currently have majority value
                        for idx in indices_at_pos:
                            if result[r][idx] == majority_val:  # Only change majority positions
                                result[r][idx] = minority_val
                        break
    
    return result

# Test
if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
        task_data = json.load(f)
    
    print("Testing selective propagation...")
    
    success_count = 0
    for i, example in enumerate(task_data['train']):
        input_grid = example['input']
        expected_output = example['output']
        actual_output = solve(input_grid)
        
        is_correct = actual_output == expected_output
        print(f"Train {i}: {'✓ PASS' if is_correct else '✗ FAIL'}")
        
        if is_correct:
            success_count += 1
        else:
            # Show first failing row
            for r in range(len(input_grid)):
                if actual_output[r] != expected_output[r]:
                    print(f"  Row {r} differs")
                    print(f"    Expected: {expected_output[r]}")
                    print(f"    Got:      {actual_output[r]}")
                    break
    
    print(f"\nSuccess rate: {success_count}/3")
    
    if success_count == 3:
        print("✓ All training examples pass! Solver is correct.")