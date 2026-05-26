#!/usr/bin/env python3

import json
import numpy as np

def detailed_analysis():
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        task = json.load(f)
    
    print("=== DETAILED ROTATION + CROP ANALYSIS ===")
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\n--- TRAIN {i} ---")
        print(f"Input shape: {input_grid.shape}")
        print(f"Output shape: {output_grid.shape}")
        
        # Test all 4 rotations
        found_match = False
        for k in range(4):
            rotated = np.rot90(input_grid, k=k)
            rotation_degrees = k * 90
            
            print(f"\nTrying {rotation_degrees}° rotation (k={k}):")
            print(f"  Rotated shape: {rotated.shape}")
            
            # Check if output is exactly the rotated input
            if rotated.shape == output_grid.shape and np.array_equal(rotated, output_grid):
                print(f"  ✓ EXACT MATCH: output = input rotated {rotation_degrees}°")
                found_match = True
                break
            
            # Check if output is a crop of the rotated input
            if (rotated.shape[0] >= output_grid.shape[0] and 
                rotated.shape[1] >= output_grid.shape[1]):
                
                for start_r in range(rotated.shape[0] - output_grid.shape[0] + 1):
                    for start_c in range(rotated.shape[1] - output_grid.shape[1] + 1):
                        end_r = start_r + output_grid.shape[0]
                        end_c = start_c + output_grid.shape[1]
                        
                        crop = rotated[start_r:end_r, start_c:end_c]
                        if np.array_equal(crop, output_grid):
                            print(f"  ✓ CROP MATCH: output = input rotated {rotation_degrees}°, cropped [{start_r}:{end_r}, {start_c}:{end_c}]")
                            found_match = True
                            break
                    
                    if found_match:
                        break
            
            if found_match:
                break
        
        if not found_match:
            print("  ✗ No rotation + crop match found")
    
    print(f"\n=== TEST CASES ===")
    for i, test_case in enumerate(task['test']):
        input_grid = np.array(test_case['input'])
        print(f"Test {i}: {input_grid.shape}")

if __name__ == "__main__":
    detailed_analysis()