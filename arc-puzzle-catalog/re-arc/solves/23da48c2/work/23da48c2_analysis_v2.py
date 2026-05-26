#!/usr/bin/env python3

import json
import numpy as np
from collections import Counter

def analyze_task_deeper():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        data = json.load(f)
    
    for i, pair in enumerate(data['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\n=== Training Pair {i+1} ===")
        print(f"Input shape: {input_grid.shape}")
        print(f"Output shape: {output_grid.shape}")
        
        # Find background color
        input_flat = input_grid.flatten()
        background_color = Counter(input_flat).most_common(1)[0][0]
        print(f"Background color: {background_color}")
        
        # Check if output has same height as input
        print(f"Height preserved: {input_grid.shape[0] == output_grid.shape[0]}")
        
        # Look for patterns in the height differences
        height_diff = input_grid.shape[0] - output_grid.shape[0]
        width_diff = input_grid.shape[1] - output_grid.shape[1]
        print(f"Height difference: {height_diff}")
        print(f"Width difference: {width_diff}")
        
        # Analyze what's in each row/column
        print("\nInput analysis:")
        non_bg_cols = []
        for col in range(input_grid.shape[1]):
            if np.any(input_grid[:, col] != background_color):
                non_bg_cols.append(col)
        print(f"Columns with non-background: {len(non_bg_cols)}/{input_grid.shape[1]}")
        print(f"First few non-bg columns: {non_bg_cols[:10]}")
        
        non_bg_rows = []
        for row in range(input_grid.shape[0]):
            if np.any(input_grid[row, :] != background_color):
                non_bg_rows.append(row)
        print(f"Rows with non-background: {len(non_bg_rows)}/{input_grid.shape[0]}")
        print(f"Non-bg rows: {non_bg_rows}")
        
        # Check if output dimensions relate to non-bg regions
        print(f"Output width vs non-bg columns: {output_grid.shape[1]} vs {len(non_bg_cols)}")
        print(f"Output height vs non-bg rows: {output_grid.shape[0]} vs {len(non_bg_rows)}")

if __name__ == "__main__":
    analyze_task_deeper()