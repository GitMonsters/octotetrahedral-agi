#!/usr/bin/env python3

import json
import numpy as np

def analyze_exact_transformation():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        task = json.load(f)
    
    print("=== FINAL ANALYSIS: EXACT TRANSFORMATION ===")
    print("Key insight: Rows stay same, only columns are reduced")
    print("This must be a 90° rotation followed by cropping to get original row count back")
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\\nTrain {i}:")
        print(f"  Input: {input_grid.shape}")
        print(f"  Output: {output_grid.shape}")
        
        # Try 90° CCW rotation
        rotated_ccw = np.rot90(input_grid, k=1)
        print(f"  90° CCW rotated: {rotated_ccw.shape}")
        
        # We need to crop this to get shape (input_rows, output_cols)
        target_rows, target_cols = output_grid.shape
        
        if rotated_ccw.shape[0] >= target_rows and rotated_ccw.shape[1] >= target_cols:
            # Try all possible crops
            found = False
            for start_row in range(rotated_ccw.shape[0] - target_rows + 1):
                for start_col in range(rotated_ccw.shape[1] - target_cols + 1):
                    crop = rotated_ccw[start_row:start_row + target_rows, 
                                     start_col:start_col + target_cols]
                    
                    if np.array_equal(crop, output_grid):
                        print(f"  ✓ FOUND: 90° CCW + crop rows[{start_row}:{start_row + target_rows}], cols[{start_col}:{start_col + target_cols}]")
                        found = True
                        break
                if found:
                    break
            
            if not found:
                print(f"  ✗ No exact crop found")
        
        # Try 90° CW rotation  
        rotated_cw = np.rot90(input_grid, k=-1)
        print(f"  90° CW rotated: {rotated_cw.shape}")
        
        if rotated_cw.shape[0] >= target_rows and rotated_cw.shape[1] >= target_cols:
            found = False
            for start_row in range(rotated_cw.shape[0] - target_rows + 1):
                for start_col in range(rotated_cw.shape[1] - target_cols + 1):
                    crop = rotated_cw[start_row:start_row + target_rows, 
                                    start_col:start_col + target_cols]
                    
                    if np.array_equal(crop, output_grid):
                        print(f"  ✓ FOUND: 90° CW + crop rows[{start_row}:{start_row + target_rows}], cols[{start_col}:{start_col + target_cols}]")
                        found = True
                        break
                if found:
                    break

if __name__ == "__main__":
    analyze_exact_transformation()