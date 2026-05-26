#!/usr/bin/env python3

import json
import numpy as np

def test_final_hypothesis():
    """
    Based on the visualization, it looks like:
    1. Rotate input 90° counterclockwise (CCW)
    2. Take the top portion that matches output size
    
    Let me test this systematically.
    """
    
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        task = json.load(f)
    
    print("=== TESTING FINAL HYPOTHESIS ===")
    print("Hypothesis: 90° CCW rotation + top crop")
    
    def solve(grid):
        # Rotate 90° counter-clockwise
        rotated = np.rot90(grid, k=1)
        # Take top portion to match expected output size
        # For this we need to know output size, so let's try common patterns
        return rotated
    
    for i, pair in enumerate(task['train']):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        print(f"\n--- TRAIN {i} ---")
        print(f"Input: {input_grid.shape}, Expected output: {output_grid.shape}")
        
        # Test different rotations with different crops
        rotations = [
            ("0°", np.rot90(input_grid, k=0)),
            ("90° CCW", np.rot90(input_grid, k=1)),  
            ("180°", np.rot90(input_grid, k=2)),
            ("270° CCW", np.rot90(input_grid, k=3))
        ]
        
        for rot_name, rotated in rotations:
            print(f"  {rot_name}: {input_grid.shape} -> {rotated.shape}")
            
            # Try different crops if dimensions allow
            if (rotated.shape[0] >= output_grid.shape[0] and 
                rotated.shape[1] >= output_grid.shape[1]):
                
                # Test specific crop positions
                crops = [
                    ("top-left", rotated[:output_grid.shape[0], :output_grid.shape[1]]),
                    ("top-right", rotated[:output_grid.shape[0], -output_grid.shape[1]:]),
                    ("bottom-left", rotated[-output_grid.shape[0]:, :output_grid.shape[1]]),
                    ("bottom-right", rotated[-output_grid.shape[0]:, -output_grid.shape[1]:])
                ]
                
                for crop_name, crop in crops:
                    if crop.shape == output_grid.shape:
                        matches = np.sum(crop == output_grid)
                        total = crop.size
                        if matches == total:
                            print(f"    ✓ EXACT MATCH: {rot_name} + {crop_name}")
                            return True
                        elif matches / total > 0.9:
                            print(f"    ~ High match: {rot_name} + {crop_name} ({matches}/{total} = {matches/total:.1%})")
    
    return False

if __name__ == "__main__":
    test_final_hypothesis()