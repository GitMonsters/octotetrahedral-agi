#!/usr/bin/env python3
"""
ARC-AGI Task 486c32d6 - Final Correct Solution

Pattern discovered through manual analysis:
1. Grid has cells separated by vertical separators (uniform columns)
2. When anomalous values exist at certain positions in SOME cells, 
   they propagate to the same positions in ALL FULL-SIZE cells
3. Partial/incomplete cells at the edge may not participate in propagation
4. The key is: find minority values at each position and propagate them
"""

def solve(grid):
    from collections import Counter
    
    if not grid or not grid[0]:
        return grid
    
    height = len(grid)
    width = len(grid[0])
    result = [row[:] for row in grid]
    
    # Find separator columns
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
    
    # Process each row
    for r in range(height):
        row = grid[r]
        
        # Extract cell contents
        cell_contents = []
        cell_positions = []
        
        for start, end in cells:
            content = []
            positions = []
            for c in range(start, end):
                if c not in separators:
                    content.append(row[c])
                    positions.append(c)
            cell_contents.append(content)
            cell_positions.append(positions)
        
        if not cell_contents:
            continue
        
        # Find the expected cell size (most common size)
        cell_sizes = [len(cell) for cell in cell_contents if cell]
        if not cell_sizes:
            continue
            
        expected_size = max(set(cell_sizes), key=cell_sizes.count)
        
        # Only consider full-size cells for pattern detection
        full_cells = []
        full_positions = []
        for i, content in enumerate(cell_contents):
            if len(content) == expected_size:
                full_cells.append(content)
                full_positions.append(cell_positions[i])
        
        if len(full_cells) < 2:
            continue
        
        # For each position in the full cell pattern
        for pos in range(expected_size):
            # Collect values at this position across full cells
            values_at_pos = []
            
            for cell in full_cells:
                values_at_pos.append(cell[pos])
            
            counter = Counter(values_at_pos)
            
            # If there are multiple values, find the minority one
            if len(counter) > 1:
                sorted_counts = counter.most_common()
                majority_val, majority_count = sorted_counts[0]
                
                # Look for minority values (anomalies)
                for val, count in sorted_counts[1:]:
                    if count < majority_count:
                        # This is an anomaly - propagate to ALL full cells
                        for pos_list in full_positions:
                            if pos < len(pos_list):
                                result[r][pos_list[pos]] = val
                        break  # Only propagate the first minority value
    
    return result

# Test the solution
if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
        task = json.load(f)
    
    print("=== TESTING FINAL CORRECTED SOLUTION ===")
    
    all_passed = True
    for i, example in enumerate(task['train']):
        input_grid = example['input']
        expected_output = example['output']
        predicted_output = solve(input_grid)
        
        is_correct = predicted_output == expected_output
        print(f"Train {i}: {'✅ PASS' if is_correct else '❌ FAIL'}")
        
        if not is_correct:
            all_passed = False
            # Show differences
            diff_rows = []
            for r in range(len(input_grid)):
                if predicted_output[r] != expected_output[r]:
                    diff_rows.append(r)
            
            print(f"  Different rows: {diff_rows[:3]}...")  # Show first 3
            for r in diff_rows[:1]:  # Show detail for first different row
                print(f"    Row {r}:")
                print(f"      Input:    {input_grid[r]}")
                print(f"      Expected: {expected_output[r]}")
                print(f"      Got:      {predicted_output[r]}")
    
    if all_passed:
        print("\n🎉 ALL TRAINING EXAMPLES PASS! 🎉")
        print("Solver is ready for submission!")
    else:
        print("\n❌ Still debugging needed...")