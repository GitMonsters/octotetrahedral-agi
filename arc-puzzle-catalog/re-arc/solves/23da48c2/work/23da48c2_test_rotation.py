#!/usr/bin/env python3

import json
import numpy as np

def test_rotation_hypothesis():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        task = json.load(f)
    
    print("=== TESTING 90° CLOCKWISE ROTATION + CROP HYPOTHESIS ===")
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\n--- TRAIN {i} ---")
        print(f"Input: {input_grid.shape}, Output: {output_grid.shape}")
        
        # 90° clockwise rotation = np.rot90(input, k=3) or k=-1
        rotated = np.rot90(input_grid, k=-1)  # 90° clockwise
        print(f"After 90° CW rotation: {rotated.shape}")
        
        # Try different crops
        found_match = False
        
        # Try all possible crops
        if (rotated.shape[0] >= output_grid.shape[0] and 
            rotated.shape[1] >= output_grid.shape[1]):
            
            for start_row in range(rotated.shape[0] - output_grid.shape[0] + 1):
                for start_col in range(rotated.shape[1] - output_grid.shape[1] + 1):
                    end_row = start_row + output_grid.shape[0]
                    end_col = start_col + output_grid.shape[1]
                    
                    crop = rotated[start_row:end_row, start_col:end_col]
                    
                    if np.array_equal(crop, output_grid):
                        print(f"  ✓ EXACT MATCH: 90°CW + crop rows[{start_row}:{end_row}] cols[{start_col}:{end_col}]")
                        found_match = True
                        # Store this for pattern recognition
                        crop_info = (start_row, end_row, start_col, end_col, rotated.shape)
                        break
                
                if found_match:
                    break
        
        if not found_match:
            # Check some common crop patterns
            crops_to_try = [
                ("top-left", rotated[:output_grid.shape[0], :output_grid.shape[1]]),
                ("bottom-right", rotated[-output_grid.shape[0]:, -output_grid.shape[1]:]),
                ("top-right", rotated[:output_grid.shape[0], -output_grid.shape[1]:]),
                ("bottom-left", rotated[-output_grid.shape[0]:, :output_grid.shape[1]])
            ]
            
            for crop_name, crop in crops_to_try:
                matches = np.sum(crop == output_grid)
                total = crop.size
                match_pct = matches / total
                print(f"  {crop_name} crop: {matches}/{total} = {match_pct:.1%}")
                
                if match_pct > 0.95:  # Very close match
                    print(f"  ✓ VERY CLOSE: {crop_name} crop")

if __name__ == "__main__":
    test_rotation_hypothesis()