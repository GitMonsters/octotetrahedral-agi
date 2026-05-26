#!/usr/bin/env python3
"""
Ultra-precise analysis of what exactly changes
"""
import json
import numpy as np

def analyze_exact_changes():
    with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
        task = json.load(f)
    
    for i, pair in enumerate(task['train']):
        print(f"\n{'='*60}")
        print(f"TRAIN PAIR {i+1} - EXACT CHANGE ANALYSIS")
        print(f"{'='*60}")
        
        input_arr = np.array(pair['input'])
        output_arr = np.array(pair['output'])
        
        # Find background
        unique, counts = np.unique(input_arr, return_counts=True)
        bg_color = unique[np.argmax(counts)]
        
        # Find all pattern cells
        pattern_cells = []
        for r in range(len(pair['input'])):
            for c in range(len(pair['input'][0])):
                if input_arr[r, c] != bg_color:
                    pattern_cells.append((r, c, input_arr[r, c]))
        
        print(f"Background color: {bg_color}")
        print(f"Pattern cells ({len(pattern_cells)}):")
        
        # Group pattern cells by rows to see structure
        by_row = {}
        for r, c, val in pattern_cells:
            if r not in by_row:
                by_row[r] = []
            by_row[r].append((c, val))
        
        for r in sorted(by_row.keys()):
            cols = sorted(by_row[r])
            print(f"  Row {r}: {cols}")
        
        # Find changes
        changes = []
        for r in range(len(pair['input'])):
            for c in range(len(pair['input'][0])):
                if input_arr[r, c] != output_arr[r, c]:
                    changes.append((r, c, input_arr[r, c], output_arr[r, c]))
        
        print(f"\nChanges ({len(changes)}):")
        for r, c, old, new in changes:
            print(f"  ({r}, {c}): {old} -> {new}")
        
        if len(changes) == 0:
            continue
        
        # Find pattern bounding box
        pattern_rs = [r for r, c, val in pattern_cells]
        pattern_cs = [c for r, c, val in pattern_cells]
        min_r, max_r = min(pattern_rs), max(pattern_rs)
        min_c, max_c = min(pattern_cs), max(pattern_cs)
        center_r = (min_r + max_r) / 2.0
        
        print(f"\nPattern bounding box: [{min_r}, {max_r}] x [{min_c}, {max_c}]")
        print(f"Pattern center row: {center_r}")
        
        # Analyze each change in detail
        print(f"\nDetailed change analysis:")
        for r, c, old, new in changes:
            print(f"\n  Change at ({r}, {c}): {old} -> {new}")
            
            # Distance from pattern center
            dist_from_center = abs(r - center_r)
            print(f"    Distance from center row: {dist_from_center:.1f}")
            
            # Vertical reflection position
            reflected_r = 2 * center_r - r
            reflected_r_int = round(reflected_r)
            print(f"    Vertical reflection at row: {reflected_r} (rounded: {reflected_r_int})")
            
            # Check what's at the reflected position
            if 0 <= reflected_r_int < len(pair['input']):
                reflected_value = input_arr[reflected_r_int, c]
                is_pattern = reflected_value != bg_color
                print(f"    Value at reflection ({reflected_r_int}, {c}): {reflected_value} (pattern: {is_pattern})")
            else:
                print(f"    Reflection position out of bounds")
            
            # Count neighbors of different types
            pattern_neighbors = 0
            bg_neighbors = 0
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(pair['input']) and 0 <= nc < len(pair['input'][0]):
                    if input_arr[nr, nc] == bg_color:
                        bg_neighbors += 1
                    else:
                        pattern_neighbors += 1
            
            print(f"    Neighbors: {pattern_neighbors} pattern, {bg_neighbors} background")

if __name__ == '__main__':
    analyze_exact_changes()