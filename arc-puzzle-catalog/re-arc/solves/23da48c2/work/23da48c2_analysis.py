#!/usr/bin/env python3

import json
import numpy as np
from collections import Counter

def analyze_task():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        data = json.load(f)
    
    for i, pair in enumerate(data['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\n=== Training Pair {i+1} ===")
        print(f"Input shape: {input_grid.shape}")
        print(f"Output shape: {output_grid.shape}")
        
        # Find background color (most frequent)
        input_flat = input_grid.flatten()
        background_color = Counter(input_flat).most_common(1)[0][0]
        print(f"Background color: {background_color}")
        
        # Find non-background pixels
        non_bg_positions = np.where(input_grid != background_color)
        if len(non_bg_positions[0]) > 0:
            min_row, max_row = non_bg_positions[0].min(), non_bg_positions[0].max()
            min_col, max_col = non_bg_positions[1].min(), non_bg_positions[1].max()
            print(f"Non-background bounding box: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
            print(f"Bounding box size: {max_row-min_row+1} x {max_col-min_col+1}")
        
        # Check if output matches cropped region
        cropped = input_grid[min_row:max_row+1, min_col:max_col+1]
        print(f"Cropped region shape: {cropped.shape}")
        
        if cropped.shape == output_grid.shape:
            matches = np.array_equal(cropped, output_grid)
            print(f"Direct crop matches output: {matches}")
            
            if not matches:
                print("Differences found - analyzing...")
                diff_positions = np.where(cropped != output_grid)
                print(f"Number of differing cells: {len(diff_positions[0])}")
                for j in range(min(5, len(diff_positions[0]))):  # Show first 5 differences
                    r, c = diff_positions[0][j], diff_positions[1][j]
                    print(f"  Position ({r},{c}): cropped={cropped[r,c]}, output={output_grid[r,c]}")
        else:
            print(f"Shape mismatch: cropped {cropped.shape} vs output {output_grid.shape}")

if __name__ == "__main__":
    analyze_task()