#!/usr/bin/env python3

import json
import numpy as np

def analyze_task():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        task = json.load(f)
    
    print("=== ARC TASK 23da48c2 DETAILED COLUMN ANALYSIS ===")
    print()
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"TRAIN {i}:")
        print(f"  Input shape:  {input_grid.shape} -> Output shape: {output_grid.shape}")
        print(f"  Columns: {input_grid.shape[1]} -> {output_grid.shape[1]} (removed {input_grid.shape[1] - output_grid.shape[1]})")
        
        # Show first few rows to understand the transformation
        print("  Input first 5 rows:")
        for row in range(min(5, input_grid.shape[0])):
            print(f"    {list(input_grid[row, :])}")
        
        print("  Output first 5 rows:")
        for row in range(min(5, output_grid.shape[0])):
            print(f"    {list(output_grid[row, :])}")
        
        # Try to find rotation/transformation
        print("  Checking if output is rotated input...")
        
        # Check 90-degree rotations
        for rotation in [0, 90, 180, 270]:
            if rotation == 0:
                rotated = input_grid
            elif rotation == 90:
                rotated = np.rot90(input_grid, k=1)
            elif rotation == 180:
                rotated = np.rot90(input_grid, k=2)
            elif rotation == 270:
                rotated = np.rot90(input_grid, k=3)
            
            # Check if any subregion of rotated matches output
            if rotated.shape == output_grid.shape and np.array_equal(rotated, output_grid):
                print(f"    EXACT MATCH: {rotation}° rotation")
                break
            
            # Check if output is a crop of the rotated input
            for start_r in range(max(1, rotated.shape[0] - output_grid.shape[0] + 1)):
                for start_c in range(max(1, rotated.shape[1] - output_grid.shape[1] + 1)):
                    end_r = start_r + output_grid.shape[0]
                    end_c = start_c + output_grid.shape[1]
                    if (end_r <= rotated.shape[0] and end_c <= rotated.shape[1] and
                        np.array_equal(rotated[start_r:end_r, start_c:end_c], output_grid)):
                        print(f"    MATCH: {rotation}° rotation + crop [{start_r}:{end_r}, {start_c}:{end_c}]")
                        break
                else:
                    continue
                break
        
        print()
    
    print("=== Test cases ===")
    for i, test_case in enumerate(task['test']):
        input_grid = np.array(test_case['input'])
        print(f"Test {i}: {input_grid.shape}")

if __name__ == "__main__":
    analyze_task()