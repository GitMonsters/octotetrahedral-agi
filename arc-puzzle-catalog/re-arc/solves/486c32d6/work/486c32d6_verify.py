#!/usr/bin/env python3
import json
from collections import Counter

# Load task
with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task = json.load(f)

def verify_hypothesis(example_idx):
    example = task['train'][example_idx]
    input_grid = example['input']
    output_grid = example['output']
    
    height = len(input_grid)
    width = len(input_grid[0])
    
    print(f"\n=== VERIFYING TRAIN {example_idx} ===")
    
    # Find separator columns
    separators = []
    for c in range(width):
        col_values = [input_grid[r][c] for r in range(height)]
        if len(set(col_values)) == 1:
            separators.append(c)
    
    print(f"Separators: {separators}")
    
    # Define cell boundaries
    cell_boundaries = [0] + [s + 1 for s in separators] + [width]
    cells = []
    for i in range(len(cell_boundaries) - 1):
        cells.append((cell_boundaries[i], cell_boundaries[i+1]))
    
    print(f"Cell ranges: {cells}")
    
    # For each row, check if anomalies were propagated
    for r in range(height):
        row_input = input_grid[r]
        row_output = output_grid[r]
        
        # Skip if no changes in this row
        if row_input == row_output:
            continue
            
        print(f"\nRow {r}:")
        
        # Find anomalies in each cell
        anomalies = []
        for cell_start, cell_end in cells:
            cell_values = []
            cell_positions = []
            for c in range(cell_start, cell_end):
                if c not in separators:
                    cell_values.append(row_input[c])
                    cell_positions.append(c)
            
            if not cell_values:
                continue
                
            # Count values in this cell
            counter = Counter(cell_values)
            
            # Find anomalies (values that appear only once)
            for pos_idx, pos in enumerate(cell_positions):
                value = row_input[pos]
                if counter[value] == 1:  # Anomaly
                    # Find the relative position within the cell
                    rel_pos = pos_idx
                    anomalies.append((rel_pos, value))
                    print(f"  Cell [{cell_start}:{cell_end}] has anomaly at rel_pos {rel_pos}: value {value}")
        
        # Check if anomalies were propagated to all cells
        if anomalies:
            print(f"  Anomalies to propagate: {anomalies}")
            
            # Verify propagation
            for cell_start, cell_end in cells:
                cell_positions = []
                for c in range(cell_start, cell_end):
                    if c not in separators:
                        cell_positions.append(c)
                
                for rel_pos, anomaly_value in anomalies:
                    if rel_pos < len(cell_positions):
                        expected_pos = cell_positions[rel_pos]
                        expected_value = anomaly_value
                        actual_value = row_output[expected_pos]
                        
                        print(f"    Cell [{cell_start}:{cell_end}], pos {expected_pos}: expected {expected_value}, got {actual_value}")
                        
                        if actual_value != expected_value:
                            print(f"    ❌ MISMATCH!")
                        else:
                            print(f"    ✓ Correct")

# Verify all training examples
for i in range(len(task['train'])):
    verify_hypothesis(i)