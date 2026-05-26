#!/usr/bin/env python3
"""
Manual verification of the vertical reflection hypothesis
"""
import json
import numpy as np

def load_task(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def verify_hypothesis(pair_num, input_grid, output_grid):
    print(f"\n=== VERIFYING TRAIN PAIR {pair_num} ===")
    
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    h, w = input_arr.shape
    
    # Find background color
    unique, counts = np.unique(input_arr, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    print(f"Background color: {bg_color}")
    
    if bg_color == 3:
        print("Background is already 3, no changes expected")
        changes = np.where(input_arr != output_arr)
        print(f"Actual changes: {len(changes[0])}")
        return len(changes[0]) == 0
    
    # Find pattern cells
    pattern_mask = input_arr != bg_color
    pattern_cells = set()
    if np.any(pattern_mask):
        pattern_positions = np.where(pattern_mask)
        pattern_cells = set(zip(pattern_positions[0], pattern_positions[1]))
        
        min_r = min(r for r, c in pattern_cells)
        max_r = max(r for r, c in pattern_cells)
        center_r = (min_r + max_r) / 2.0
        print(f"Pattern rows: [{min_r}, {max_r}], center_r: {center_r}")
    
    # Check each cell to see if hypothesis predicts correctly
    correct_predictions = 0
    total_predictions = 0
    
    for r in range(h):
        for c in range(w):
            if input_arr[r, c] == bg_color:  # Background cell
                # Calculate vertical reflection
                reflected_r = 2 * center_r - r
                reflected_r_int = round(reflected_r)
                
                # Should this cell become 3?
                should_be_3 = (0 <= reflected_r_int < h and 
                             (reflected_r_int, c) in pattern_cells)
                
                # What actually happened?
                actually_3 = (output_arr[r, c] == 3)
                
                total_predictions += 1
                if should_be_3 == actually_3:
                    correct_predictions += 1
                else:
                    print(f"  MISMATCH at ({r}, {c}): predicted {should_be_3}, actual {actually_3}")
                    print(f"    Reflected to ({reflected_r_int}, {c}), in pattern: {(reflected_r_int, c) in pattern_cells}")
    
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.3f}")
    return accuracy == 1.0

def main():
    task = load_task('/Users/evanpieser/apr12_tasks/4e600a86.json')
    
    all_correct = True
    for i, pair in enumerate(task['train']):
        correct = verify_hypothesis(i + 1, pair['input'], pair['output'])
        all_correct = all_correct and correct
        
    print(f"\n=== OVERALL RESULT ===")
    print(f"Hypothesis correct for all training pairs: {all_correct}")

if __name__ == '__main__':
    main()