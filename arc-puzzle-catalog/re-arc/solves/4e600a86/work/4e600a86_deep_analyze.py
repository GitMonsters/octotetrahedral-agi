#!/usr/bin/env python3
"""
More detailed analysis to understand the exact transformation rule
"""
import json
import numpy as np

def load_task(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_changes_deeply(pair_num, input_grid, output_grid):
    print(f"\n=== DEEP ANALYSIS TRAIN PAIR {pair_num} ===")
    
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    h, w = input_arr.shape
    
    # Find background and changes
    unique, counts = np.unique(input_arr, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    changes = np.where(input_arr != output_arr)
    change_positions = set(zip(changes[0], changes[1]))
    
    if bg_color == 3 or len(change_positions) == 0:
        print("No changes to analyze")
        return
    
    # Find pattern cells
    pattern_mask = input_arr != bg_color
    pattern_positions = np.where(pattern_mask)
    pattern_cells = set(zip(pattern_positions[0], pattern_positions[1]))
    
    min_r = min(r for r, c in pattern_cells)
    max_r = max(r for r, c in pattern_cells)
    min_c = min(c for r, c in pattern_cells)  
    max_c = max(c for r, c in pattern_cells)
    center_r = (min_r + max_r) / 2.0
    
    print(f"Pattern bounding box: [{min_r}, {max_r}] x [{min_c}, {max_c}]")
    print(f"Center row: {center_r}")
    
    # Analyze what makes a background cell eligible to become 3
    print("Analyzing changes:")
    for r, c in sorted(change_positions):
        reflected_r = 2 * center_r - r
        reflected_r_int = round(reflected_r)
        
        has_reflection = (0 <= reflected_r_int < h and 
                         (reflected_r_int, c) in pattern_cells)
        
        # Check if position is within bounding box
        in_bbox_row = min_r <= r <= max_r
        in_bbox_col = min_c <= c <= max_c
        in_bbox = in_bbox_row and in_bbox_col
        
        print(f"  Change ({r}, {c}): reflection at ({reflected_r_int}, {c}) exists: {has_reflection}, in bbox: {in_bbox}")
    
    # Check what background cells DON'T change even though they have reflections
    print("Background cells with reflections that DON'T change:")
    count_missed = 0
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if input_arr[r, c] == bg_color and (r, c) not in change_positions:
                reflected_r = 2 * center_r - r
                reflected_r_int = round(reflected_r)
                
                has_reflection = (0 <= reflected_r_int < h and 
                                (reflected_r_int, c) in pattern_cells)
                
                if has_reflection:
                    count_missed += 1
                    if count_missed <= 10:  # Show first 10
                        print(f"  NO change ({r}, {c}): reflection at ({reflected_r_int}, {c}) exists")
    
    if count_missed > 10:
        print(f"  ... and {count_missed - 10} more")

def main():
    task = load_task('/Users/evanpieser/apr12_tasks/4e600a86.json')
    
    for i, pair in enumerate(task['train']):
        analyze_changes_deeply(i + 1, pair['input'], pair['output'])

if __name__ == '__main__':
    main()