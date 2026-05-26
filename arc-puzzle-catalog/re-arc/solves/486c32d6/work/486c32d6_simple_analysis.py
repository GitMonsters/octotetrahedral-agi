#!/usr/bin/env python3
import json
from collections import Counter

# Load task
with open('/Users/evanpieser/apr12_tasks/486c32d6.json', 'r') as f:
    task = json.load(f)

def analyze_simple_pattern():
    print("=== SIMPLE PATTERN ANALYSIS ===")
    
    for i, example in enumerate(task['train']):
        print(f"\nTRAIN {i}:")
        input_grid = example['input']
        output_grid = example['output']
        
        # Find rows that changed
        changed_rows = []
        for r in range(len(input_grid)):
            if input_grid[r] != output_grid[r]:
                changed_rows.append(r)
        
        print(f"Changed rows: {changed_rows}")
        
        for row_idx in changed_rows:
            print(f"\nRow {row_idx}:")
            input_row = input_grid[row_idx]
            output_row = output_grid[row_idx]
            
            print(f"Input:  {input_row}")
            print(f"Output: {output_row}")
            
            # Find differences 
            diffs = []
            for c in range(len(input_row)):
                if input_row[c] != output_row[c]:
                    diffs.append((c, input_row[c], output_row[c]))
            
            print(f"Changes: {diffs}")
            
            # Look for pattern: find unique values that got propagated
            counter = Counter(input_row)
            
            # Find values that appear only once or few times
            rare_values = []
            for val, count in counter.items():
                if count < 3:  # Consider as rare/anomalous
                    rare_values.append(val)
            
            print(f"Rare values in input: {rare_values}")
            
            # Check if these rare values got propagated
            output_counter = Counter(output_row)
            print(f"Value counts in input:  {dict(counter)}")
            print(f"Value counts in output: {dict(output_counter)}")

analyze_simple_pattern()