#!/usr/bin/env python3

import json
import numpy as np
from collections import Counter

def test_column_compression():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        data = json.load(f)
    
    for i, pair in enumerate(data['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\n=== Training Pair {i+1} ===")
        
        # Find background color
        input_flat = input_grid.flatten()
        background_color = Counter(input_flat).most_common(1)[0][0]
        
        # Find non-empty columns
        non_empty_cols = []
        for col in range(input_grid.shape[1]):
            if np.any(input_grid[:, col] != background_color):
                non_empty_cols.append(col)
        
        print(f"Non-empty columns: {non_empty_cols}")
        print(f"Number of non-empty columns: {len(non_empty_cols)}")
        print(f"Output width: {output_grid.shape[1]}")
        print(f"Match: {len(non_empty_cols) == output_grid.shape[1]}")
        
        # Test: extract only non-empty columns and see if it matches output
        if len(non_empty_cols) == output_grid.shape[1]:
            compressed = input_grid[:, non_empty_cols]
            print(f"Compressed shape: {compressed.shape}")
            print(f"Compressed matches output: {np.array_equal(compressed, output_grid)}")
            
            if not np.array_equal(compressed, output_grid):
                print("Analyzing differences...")
                diff_positions = np.where(compressed != output_grid)
                print(f"Number of differences: {len(diff_positions[0])}")
                # Show some examples
                for j in range(min(3, len(diff_positions[0]))):
                    r, c = diff_positions[0][j], diff_positions[1][j]
                    print(f"  Position ({r},{c}): compressed={compressed[r,c]}, output={output_grid[r,c]}")

if __name__ == "__main__":
    test_column_compression()