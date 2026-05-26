#!/usr/bin/env python3
import json
from collections import Counter

def solve(grid):
    """
    Solve the ARC-AGI task 486c32d6.
    
    Pattern: The grid is divided into repeating cells by separator columns. 
    Anomalous values in rows get propagated to corresponding positions in all cells.
    """
    if not grid or not grid[0]:
        return grid
    
    height = len(grid)
    width = len(grid[0])
    
    # Find separator columns (columns with uniform values)
    separators = []
    for c in range(width):
        col_values = [grid[r][c] for r in range(height)]
        if len(set(col_values)) == 1:
            separators.append(c)
    
    # If no separators found, return original grid
    if not separators:
        return [row[:] for row in grid]
    
    # Define cell boundaries
    cell_boundaries = [0] + [s + 1 for s in separators] + [width]
    cells = []
    for i in range(len(cell_boundaries) - 1):
        cells.append((cell_boundaries[i], cell_boundaries[i+1]))
    
    # Create output grid as copy of input
    result = [row[:] for row in grid]
    
    # Process each row
    for r in range(height):
        row = grid[r]
        
        # Get non-separator positions and values for this row
        non_sep_positions = []
        non_sep_values = []
        for c in range(width):
            if c not in separators:
                non_sep_positions.append(c)
                non_sep_values.append(row[c])
        
        if not non_sep_values:
            continue
            
        # Find anomalous values across all cells
        # Collect values from each cell and find patterns
        all_anomalies = {}  # position_in_cell -> anomaly_value
        
        for cell_start, cell_end in cells:
            # Get values for this cell (excluding separators)
            cell_positions = []
            cell_values = []
            for c in range(cell_start, cell_end):
                if c not in separators:
                    cell_positions.append(c)
                    cell_values.append(row[c])
            
            if not cell_values:
                continue
                
            # Count values in this cell
            counter = Counter(cell_values)
            
            # Find anomalous values (appear only once)
            for pos_idx, pos in enumerate(cell_positions):
                value = row[pos]
                if counter[value] == 1:
                    # This is an anomaly - store its relative position
                    all_anomalies[pos_idx] = value
        
        # Propagate anomalies to all cells
        if all_anomalies:
            for cell_start, cell_end in cells:
                # Get positions for this cell
                cell_positions = []
                for c in range(cell_start, cell_end):
                    if c not in separators:
                        cell_positions.append(c)
                
                # Apply anomalies
                for rel_pos, anomaly_value in all_anomalies.items():
                    if rel_pos < len(cell_positions):
                        result[r][cell_positions[rel_pos]] = anomaly_value
    
    return result

# Test the solution
if __name__ == "__main__":
    # Load task
    with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
        task = json.load(f)
    
    print("=== TESTING SOLVER ===")
    
    # Test on all training examples
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
            
            # Show differences for debugging
            for r in range(len(input_grid)):
                if predicted_output[r] != expected_output[r]:
                    print(f"  Row {r} differs:")
                    print(f"    Expected: {expected_output[r]}")
                    print(f"    Got:      {predicted_output[r]}")
    
    if all_passed:
        print("\n🎉 All training examples passed!")
    else:
        print("\n⚠️ Some training examples failed - need debugging")