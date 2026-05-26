#!/usr/bin/env python3
import json

# Load task
with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task = json.load(f)

print("=== COMPREHENSIVE PATTERN ANALYSIS ===")

def analyze_row_pattern(example_idx, row_idx):
    example = task['train'][example_idx]
    input_grid = example['input']
    output_grid = example['output']
    
    input_row = input_grid[row_idx]
    output_row = output_grid[row_idx]
    
    if input_row == output_row:
        return  # No changes
    
    print(f"\nTrain {example_idx}, Row {row_idx}:")
    print(f"Input:  {input_row}")
    print(f"Output: {output_row}")
    
    # Find differences
    diffs = []
    for c in range(len(input_row)):
        if input_row[c] != output_row[c]:
            diffs.append((c, input_row[c], output_row[c]))
    
    print(f"Changes: {diffs}")
    
    # Look for the source pattern
    print("Looking for anomalous values that got propagated...")
    
    # Find all unique values in the row
    unique_vals = set(input_row)
    print(f"Unique values in input: {unique_vals}")
    
    # For each change, try to find where the new value came from
    for pos, old_val, new_val in diffs:
        print(f"  Change at pos {pos}: {old_val} -> {new_val}")
        # Find where this new_val appears in the input
        sources = [i for i, v in enumerate(input_row) if v == new_val]
        print(f"    Value {new_val} appears at positions: {sources}")

# Analyze all changed rows
for i in range(3):  # 3 training examples
    example = task['train'][i]
    input_grid = example['input']
    output_grid = example['output']
    
    # Find changed rows
    for r in range(len(input_grid)):
        if input_grid[r] != output_grid[r]:
            analyze_row_pattern(i, r)