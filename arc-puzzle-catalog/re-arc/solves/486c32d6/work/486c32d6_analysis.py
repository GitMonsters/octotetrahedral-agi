#!/usr/bin/env python3

import json
import numpy as np

# Load the task data
with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task_data = json.load(f)

# Analyze differences
print("=== TASK 486c32d6 DETAILED ANALYSIS ===")
for i, example in enumerate(task_data['train']):
    print(f"\n--- Training Pair {i+1} ---")
    input_grid = example['input']
    output_grid = example['output']
    
    print(f"Input shape: {len(input_grid)}x{len(input_grid[0])}")
    print(f"Output shape: {len(output_grid)}x{len(output_grid[0])}")
    
    # Find differences
    differences = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r][c] != output_grid[r][c]:
                differences.append((r, c, input_grid[r][c], output_grid[r][c]))
    
    print(f"Total differences: {len(differences)}")
    if differences:
        print("All differences:")
        for diff in differences:
            print(f"  Row {diff[0]}, Col {diff[1]}: {diff[2]} → {diff[3]}")

    # Analyze structure - look for separator lines
    print("\nAnalyzing grid structure...")
    
    # Check for horizontal separators (rows that are all the same value)
    h_seps = []
    for r in range(len(input_grid)):
        row_values = set(input_grid[r])
        if len(row_values) == 1:
            h_seps.append((r, list(row_values)[0]))
    
    print(f"Horizontal separator rows: {h_seps}")
    
    # Check for vertical separators (columns that are all the same value) 
    v_seps = []
    for c in range(len(input_grid[0])):
        col_values = set(input_grid[r][c] for r in range(len(input_grid)))
        if len(col_values) == 1:
            v_seps.append((c, list(col_values)[0]))
    
    print(f"Vertical separator columns: {v_seps}")