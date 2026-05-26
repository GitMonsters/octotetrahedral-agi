#!/usr/bin/env python3
"""
ARC-AGI Task 486c32d6 - DSTAR Self-Debug Solution

After extensive analysis, the pattern is:
1. Grid is divided into cells by vertical separators (uniform columns)
2. When anomalous values exist in certain cells, they propagate to corresponding positions in other cells WITHIN THE SAME ROW
3. Anomalies are values that appear in fewer cells than the majority pattern

Key insight: Look at the actual transformations - values get copied from where they exist to where they don't
"""

def solve(grid):
    from collections import Counter
    
    if not grid or not grid[0]:
        return grid
    
    height = len(grid)
    width = len(grid[0])
    result = [row[:] for row in grid]
    
    # Find separator columns (uniform values across all rows)
    separators = []
    for c in range(width):
        col_vals = [grid[r][c] for r in range(height)]
        if len(set(col_vals)) == 1:
            separators.append(c)
    
    if not separators:
        return result
    
    # Define cell boundaries
    boundaries = [0] + [s + 1 for s in separators] + [width]
    cells = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
    
    # Process each row individually
    for r in range(height):
        row = grid[r]
        
        # Extract cell contents (positions not in separators)
        cell_contents = []
        cell_positions = []
        
        for start, end in cells:
            content = []
            positions = []
            for c in range(start, end):
                if c not in separators:
                    content.append(row[c])
                    positions.append(c)
            if content:  # Only add non-empty cells
                cell_contents.append(content)
                cell_positions.append(positions)
        
        if len(cell_contents) < 2:
            continue  # Need at least 2 cells to compare
        
        cell_size = len(cell_contents[0])
        
        # For each position in the cell pattern
        for pos in range(cell_size):
            # Collect values at this position across all cells
            values_here = []
            indices_here = []
            
            for cell_idx in range(len(cell_contents)):
                if pos < len(cell_contents[cell_idx]):
                    values_here.append(cell_contents[cell_idx][pos])
                    indices_here.append(cell_positions[cell_idx][pos])
            
            # Find the pattern: if there are different values, propagate the minority one
            counter = Counter(values_here)
            
            if len(counter) > 1:  # There are differences
                # Get the counts sorted by frequency
                freq_items = counter.most_common()
                majority_val, majority_count = freq_items[0]
                
                # Find minority values that should be propagated
                for val, count in freq_items[1:]:
                    if count > 0:  # This is a minority value that exists
                        # Propagate this value to ALL positions
                        for idx in indices_here:
                            result[r][idx] = val
                        break  # Only propagate the first (most frequent) minority value
    
    return result

# Test the solution
if __name__ == "__main__":
    import json
    
    # Load the task
    with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
        task = json.load(f)
    
    print("=== TESTING DSTAR SOLUTION ===")
    
    all_passed = True
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        is_correct = predicted_output == expected_output
        print(f"Train {i}: {'✅ PASS' if is_correct else '❌ FAIL'}")
        
        if not is_correct:
            all_passed = False
            print(f"  First differing row:")
            for r in range(len(input_grid)):
                if predicted_output[r] != expected_output[r]:
                    print(f"    Row {r}:")
                    print(f"      Input:    {input_grid[r]}")
                    print(f"      Expected: {expected_output[r]}")
                    print(f"      Got:      {predicted_output[r]}")
                    break
    
    print(f"\nResult: {'🎉 ALL TESTS PASS!' if all_passed else '❌ Some tests failed'}")