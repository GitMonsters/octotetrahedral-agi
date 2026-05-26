#!/usr/bin/env python3

import json
import numpy as np

def exhaustive_search():
    """
    Try every combination of:
    - Rotation (0°, 90°, 180°, 270°)
    - Flips (none, horizontal, vertical, both)
    - Crops (different positions)
    """
    
    with open('/Users/evanpieser/apr12_tasks/23da48c2.json', 'r') as f:
        task = json.load(f)
    
    print("=== EXHAUSTIVE SEARCH FOR TRANSFORMATION ===")
    
    # Test on first training example to find the pattern
    input_grid = np.array(task['train'][0]['input'])
    output_grid = np.array(task['train'][0]['output'])
    
    print(f"Searching for: {input_grid.shape} -> {output_grid.shape}")
    
    transformations = []
    
    # Try all rotations
    for k in range(4):
        rotated = np.rot90(input_grid, k=k)
        rotation_name = f"{k*90}°"
        
        # Try all flips
        flip_ops = [
            ("none", lambda x: x),
            ("horizontal", lambda x: np.fliplr(x)),
            ("vertical", lambda x: np.flipud(x)),
            ("both", lambda x: np.flipud(np.fliplr(x)))
        ]
        
        for flip_name, flip_op in flip_ops:
            flipped = flip_op(rotated)
            
            # Try all possible crops if dimensions allow
            if (flipped.shape[0] >= output_grid.shape[0] and 
                flipped.shape[1] >= output_grid.shape[1]):
                
                for start_row in range(flipped.shape[0] - output_grid.shape[0] + 1):
                    for start_col in range(flipped.shape[1] - output_grid.shape[1] + 1):
                        end_row = start_row + output_grid.shape[0]
                        end_col = start_col + output_grid.shape[1]
                        
                        crop = flipped[start_row:end_row, start_col:end_col]
                        
                        if np.array_equal(crop, output_grid):
                            transformation = {
                                'rotation': k,
                                'flip': flip_name,
                                'crop': (start_row, end_row, start_col, end_col),
                                'description': f"Rotate {rotation_name}, flip {flip_name}, crop [{start_row}:{end_row}, {start_col}:{end_col}]"
                            }
                            transformations.append(transformation)
                            print(f"FOUND: {transformation['description']}")
    
    if transformations:
        print(f"\\nFound {len(transformations)} possible transformations")
        
        # Test the first one on all training examples
        best_transform = transformations[0]
        print(f"\\nTesting: {best_transform['description']}")
        
        def apply_transform(grid, transform):
            grid = np.array(grid)
            
            # Apply rotation
            rotated = np.rot90(grid, k=transform['rotation'])
            
            # Apply flip
            if transform['flip'] == 'horizontal':
                flipped = np.fliplr(rotated)
            elif transform['flip'] == 'vertical':
                flipped = np.flipud(rotated)
            elif transform['flip'] == 'both':
                flipped = np.flipud(np.fliplr(rotated))
            else:
                flipped = rotated
            
            # Apply crop
            start_row, end_row, start_col, end_col = transform['crop']
            
            # Need to adjust crop for different input sizes
            # For now, try proportional cropping
            if flipped.shape == (28, 22):  # Same as training example 0
                result = flipped[start_row:end_row, start_col:end_col]
            else:
                # Proportional crop
                rows_ratio = (end_row - start_row) / 28
                cols_ratio = (end_col - start_col) / 22
                
                new_rows = int(flipped.shape[0] * rows_ratio)
                new_cols = int(flipped.shape[1] * cols_ratio)
                
                result = flipped[:new_rows, :new_cols]
            
            return result
        
        # Test on all training examples
        for i, pair in enumerate(task['train']):
            try:
                predicted = apply_transform(pair['input'], best_transform)
                expected = np.array(pair['output'])
                
                if predicted.shape == expected.shape:
                    matches = np.sum(predicted == expected)
                    total = expected.size
                    print(f"Train {i}: {matches}/{total} = {matches/total:.1%}")
                else:
                    print(f"Train {i}: Shape mismatch - {predicted.shape} vs {expected.shape}")
            except Exception as e:
                print(f"Train {i}: Error - {e}")
    
    else:
        print("No transformations found!")

if __name__ == "__main__":
    exhaustive_search()