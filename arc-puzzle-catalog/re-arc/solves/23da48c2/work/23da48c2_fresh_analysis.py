#!/usr/bin/env python3

import json
import numpy as np

def find_transformation_rule():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        task = json.load(f)
    
    print("=== FRESH ANALYSIS: LOOKING FOR ACTUAL TRANSFORMATION ===")
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\\n--- TRAIN {i} ---")
        print(f"Input: {input_grid.shape}, Output: {output_grid.shape}")
        
        # Let's examine the structure more carefully
        # Look for non-background patterns in input
        input_bg = 7 if i < 3 else 4  # Background is 7 for first 3, 4 for last
        
        # Find bounding box of non-background in input
        non_bg_rows, non_bg_cols = np.where(input_grid != input_bg)
        if len(non_bg_rows) > 0:
            min_row, max_row = non_bg_rows.min(), non_bg_rows.max()
            min_col, max_col = non_bg_cols.min(), non_bg_cols.max()
            print(f"  Non-background region: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
            print(f"  Region size: {max_row-min_row+1} x {max_col-min_col+1}")
        
        # Find bounding box of non-background in output
        non_bg_rows_out, non_bg_cols_out = np.where(output_grid != input_bg)
        if len(non_bg_rows_out) > 0:
            min_row_out, max_row_out = non_bg_rows_out.min(), non_bg_rows_out.max()
            min_col_out, max_col_out = non_bg_cols_out.min(), non_bg_cols_out.max()
            print(f"  Output non-bg region: rows {min_row_out}-{max_row_out}, cols {min_col_out}-{max_col_out}")
            print(f"  Output region size: {max_row_out-min_row_out+1} x {max_col_out-min_col_out+1}")
        
        # Check if input rotated 90° CW and then cropped matches output
        # 90° clockwise = 270° counter-clockwise = np.rot90(input, k=3)
        rotated_cw = np.rot90(input_grid, k=3)
        print(f"  Input rotated 90° CW: {rotated_cw.shape}")
        
        # Try cropping the rotated input to match output size
        if (rotated_cw.shape[0] >= output_grid.shape[0] and 
            rotated_cw.shape[1] >= output_grid.shape[1]):
            
            found_exact = False
            for start_row in range(rotated_cw.shape[0] - output_grid.shape[0] + 1):
                for start_col in range(rotated_cw.shape[1] - output_grid.shape[1] + 1):
                    end_row = start_row + output_grid.shape[0]
                    end_col = start_col + output_grid.shape[1]
                    
                    crop = rotated_cw[start_row:end_row, start_col:end_col]
                    
                    if np.array_equal(crop, output_grid):
                        print(f"  ✓ EXACT MATCH: 90°CW rotation + crop [{start_row}:{end_row}, {start_col}:{end_col}]")
                        found_exact = True
                        break
                
                if found_exact:
                    break
            
            if not found_exact:
                print(f"  ✗ No exact match found with 90°CW rotation + crop")
                
                # Let's try with the top-left crop (most common)
                crop = rotated_cw[:output_grid.shape[0], :output_grid.shape[1]]
                matches = np.sum(crop == output_grid)
                total = crop.size
                print(f"  Top-left crop match: {matches}/{total} = {matches/total:.1%}")
                
                # Let's also try bottom-right crop
                crop = rotated_cw[-output_grid.shape[0]:, -output_grid.shape[1]:]
                matches = np.sum(crop == output_grid)
                print(f"  Bottom-right crop match: {matches}/{total} = {matches/total:.1%}")
        
        # Also check 90° CCW (counter-clockwise)
        rotated_ccw = np.rot90(input_grid, k=1)
        print(f"  Input rotated 90° CCW: {rotated_ccw.shape}")
        
        if (rotated_ccw.shape[0] >= output_grid.shape[0] and 
            rotated_ccw.shape[1] >= output_grid.shape[1]):
            
            found_exact = False
            for start_row in range(rotated_ccw.shape[0] - output_grid.shape[0] + 1):
                for start_col in range(rotated_ccw.shape[1] - output_grid.shape[1] + 1):
                    end_row = start_row + output_grid.shape[0]
                    end_col = start_col + output_grid.shape[1]
                    
                    crop = rotated_ccw[start_row:end_row, start_col:end_col]
                    
                    if np.array_equal(crop, output_grid):
                        print(f"  ✓ EXACT MATCH: 90°CCW rotation + crop [{start_row}:{end_row}, {start_col}:{end_col}]")
                        found_exact = True
                        break
                
                if found_exact:
                    break
            
            if not found_exact:
                # Try top-left crop
                crop = rotated_ccw[:output_grid.shape[0], :output_grid.shape[1]]
                matches = np.sum(crop == output_grid)
                total = crop.size
                print(f"  90°CCW top-left crop match: {matches}/{total} = {matches/total:.1%}")

if __name__ == "__main__":
    find_transformation_rule()