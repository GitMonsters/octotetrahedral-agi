#!/usr/bin/env python3
"""
ARC-AGI Task 486c32d6 Solver

Final understanding from manual analysis:
- Grid has cells separated by vertical separator columns (uniform values)
- Within each row, when there are anomalous values in some cells, 
  they should be propagated to the corresponding positions in ALL cells
"""

def solve(grid):
    if not grid or not grid[0]:
        return grid
    
    height = len(grid)  
    width = len(grid[0])
    
    # Find separator columns - columns with all same value
    separators = set()
    for c in range(width):
        col_vals = [grid[r][c] for r in range(height)]
        if len(set(col_vals)) == 1:
            separators.add(c)
    
    if not separators:
        return [row[:] for row in grid]
    
    # Find cell boundaries
    separators_list = sorted(list(separators))
    cell_starts = [0] + [s + 1 for s in separators_list]
    cell_ends = separators_list + [width]
    
    cells = list(zip(cell_starts, cell_ends))
    
    result = [row[:] for row in grid]
    
    # Process each row
    for r in range(height):
        row = grid[r]
        
        # Get cell contents for this row
        cell_data = []
        for start, end in cells:
            cell_content = []
            for c in range(start, end):
                if c not in separators:
                    cell_content.append((c, row[c]))
            cell_data.append(cell_content)
        
        if not cell_data or not cell_data[0]:
            continue
            
        # Check each position in the cell pattern
        cell_size = len(cell_data[0])
        
        for pos in range(cell_size):
            # Collect values at this position across all cells
            pos_values = []
            pos_indices = []
            
            for cell in cell_data:
                if pos < len(cell):
                    idx, val = cell[pos]
                    pos_values.append(val)
                    pos_indices.append(idx)
            
            if not pos_values:
                continue
            
            # Find anomalies - values that appear rarely
            from collections import Counter
            counter = Counter(pos_values)
            
            if len(counter) > 1:
                # Find values that appear infrequently (anomalies)
                total_cells = len(pos_values)
                
                for val, count in counter.items():
                    # If this value appears in less than half the cells, it might be an anomaly
                    if count <= 2 and count < total_cells - 1:
                        # This is an anomaly - propagate it to ALL positions
                        for idx in pos_indices:
                            result[r][idx] = val
                        break  # Only propagate one anomaly per position
    
    return result

# Test the solver
if __name__ == "__main__":
    import json
    
    # Load task
    with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
        task = json.load(f)
    
    print("=== TESTING FINAL SOLVER ===")
    
    all_passed = True
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        if predicted_output == expected_output:
            print(f"✓ Train {i}: PASS")
        else:
            print(f"❌ Train {i}: FAIL")
            all_passed = False
            
            # Show first difference
            for r in range(len(input_grid)):
                if predicted_output[r] != expected_output[r]:
                    print(f"  Row {r}:")
                    print(f"    Input:    {input_grid[r]}")
                    print(f"    Expected: {expected_output[r]}")
                    print(f"    Got:      {predicted_output[r]}")
                    break
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print("\n❌ Tests failed - need more debugging")