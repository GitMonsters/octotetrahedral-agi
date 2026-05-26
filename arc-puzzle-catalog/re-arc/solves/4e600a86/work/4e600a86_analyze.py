#!/usr/bin/env python3
"""
Detailed analysis of ARC task 4e600a86 transformation patterns
"""
import json
import numpy as np

def load_task(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_pair(pair_num, input_grid, output_grid):
    print(f"\n=== DETAILED ANALYSIS TRAIN PAIR {pair_num} ===")
    
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    
    h, w = input_arr.shape
    print(f"Grid size: {h} x {w}")
    
    # Find background color (most frequent)
    unique, counts = np.unique(input_arr, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    print(f"Background color: {bg_color}")
    
    # Find pattern colors
    pattern_colors = [color for color in unique if color != bg_color]
    print(f"Pattern colors: {pattern_colors}")
    
    # Find all pattern cells
    pattern_mask = input_arr != bg_color
    pattern_cells = np.where(pattern_mask)
    pattern_positions = list(zip(pattern_cells[0], pattern_cells[1]))
    
    if pattern_positions:
        min_r, max_r = min(r for r, c in pattern_positions), max(r for r, c in pattern_positions)
        min_c, max_c = min(c for r, c in pattern_positions), max(c for r, c in pattern_positions)
        print(f"Pattern bounding box: rows [{min_r}, {max_r}], cols [{min_c}, {max_c}]")
        print(f"Pattern height: {max_r - min_r + 1}, width: {max_c - min_c + 1}")
        
        # Calculate center
        center_r = (min_r + max_r) / 2.0
        center_c = (min_c + max_c) / 2.0
        print(f"Pattern center: ({center_r:.1f}, {center_c:.1f})")
    
    # Find all changes
    changes = np.where(input_arr != output_arr)
    change_positions = list(zip(changes[0], changes[1]))
    print(f"Number of changes: {len(change_positions)}")
    
    if change_positions:
        print("Changes (row, col): old -> new")
        for r, c in change_positions:
            print(f"  ({r}, {c}): {input_arr[r, c]} -> {output_arr[r, c]}")
        
        # Analyze what changes
        changed_from = input_arr[changes]
        changed_to = output_arr[changes]
        print(f"All changes are from {set(changed_from)} to {set(changed_to)}")
        
        # Check if changes are symmetric/mirrored
        if pattern_positions:
            print("\nAnalyzing spatial relationship between changes and pattern:")
            
            for r, c in change_positions:
                # Find closest pattern cell
                distances = [(abs(r - pr) + abs(c - pc), pr, pc) for pr, pc in pattern_positions]
                min_dist, closest_pr, closest_pc = min(distances)
                print(f"  Change at ({r}, {c}) closest to pattern at ({closest_pr}, {closest_pc}), distance: {min_dist}")
                
                # Check if there's a pattern cell at the horizontally reflected position
                reflected_c = 2 * center_c - c
                if abs(reflected_c - round(reflected_c)) < 0.1:  # Close to integer
                    reflected_c_int = round(reflected_c)
                    if 0 <= reflected_c_int < w and (r, reflected_c_int) in pattern_positions:
                        print(f"    Has horizontal reflection in pattern at ({r}, {reflected_c_int})")
                
                # Check if there's a pattern cell at the vertically reflected position  
                reflected_r = 2 * center_r - r
                if abs(reflected_r - round(reflected_r)) < 0.1:  # Close to integer
                    reflected_r_int = round(reflected_r)
                    if 0 <= reflected_r_int < h and (reflected_r_int, c) in pattern_positions:
                        print(f"    Has vertical reflection in pattern at ({reflected_r_int}, {c})")

def main():
    task = load_task('/Users/evanpieser/apr12_tasks/4e600a86.json')
    
    for i, pair in enumerate(task['train']):
        analyze_pair(i + 1, pair['input'], pair['output'])
        
    print(f"\n=== TEST CASES ===")
    for i, test_case in enumerate(task['test']):
        input_arr = np.array(test_case['input'])
        h, w = input_arr.shape
        
        # Find background and pattern
        unique, counts = np.unique(input_arr, return_counts=True)
        bg_color = unique[np.argmax(counts)]
        pattern_colors = [color for color in unique if color != bg_color]
        
        print(f"Test {i+1}: Size {h}x{w}, Background: {bg_color}, Pattern: {pattern_colors}")
        
        # Find pattern bounding box
        pattern_mask = input_arr != bg_color
        if np.any(pattern_mask):
            pattern_cells = np.where(pattern_mask)
            pattern_positions = list(zip(pattern_cells[0], pattern_cells[1]))
            min_r, max_r = min(r for r, c in pattern_positions), max(r for r, c in pattern_positions)
            min_c, max_c = min(c for r, c in pattern_positions), max(c for r, c in pattern_positions)
            center_r = (min_r + max_r) / 2.0
            center_c = (min_c + max_c) / 2.0
            print(f"  Pattern bbox: [{min_r}, {max_r}] x [{min_c}, {max_c}], center: ({center_r:.1f}, {center_c:.1f})")

if __name__ == '__main__':
    main()