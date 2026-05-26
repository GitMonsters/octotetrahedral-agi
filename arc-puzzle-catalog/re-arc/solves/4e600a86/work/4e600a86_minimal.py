#!/usr/bin/env python3
"""
Minimal rule approach - try to find the simplest rule that works
"""
import json
import numpy as np
import copy

def solve_minimal(grid):
    """
    Minimal approach: Only change specific cells based on exact pattern matching
    """
    result = copy.deepcopy(grid)
    h, w = len(grid), len(grid[0])
    
    # Find background color
    colors = {}
    for row in grid:
        for cell in row:
            colors[cell] = colors.get(cell, 0) + 1
    bg_color = max(colors, key=colors.get)
    
    # If background is already 3, return unchanged
    if bg_color == 3:
        return result
    
    # Find pattern cells
    pattern_cells = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg_color:
                pattern_cells.add((r, c))
    
    if not pattern_cells:
        return result
    
    # Find bounding box
    min_r = min(r for r, c in pattern_cells)
    max_r = max(r for r, c in pattern_cells)
    min_c = min(c for r, c in pattern_cells) 
    max_c = max(c for r, c in pattern_cells)
    
    center_r = (min_r + max_r) / 2.0
    
    # Let me try a very specific rule based on the exact observations:
    # Only fill background cells where:
    # 1. They are in the bottom half of the pattern (below center)
    # 2. AND they have a vertical reflection in the pattern
    # 3. OR they have multiple pattern neighbors
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg_color and min_r <= r <= max_r and min_c <= c <= max_c:
                
                # Check if below center
                below_center = r > center_r
                
                # Check vertical reflection
                reflected_r = 2 * center_r - r
                reflected_r_int = round(reflected_r)
                has_reflection = (min_r <= reflected_r_int <= max_r and 
                                (reflected_r_int, c) in pattern_cells)
                
                # Count direct neighbors that are pattern cells
                pattern_neighbors = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in pattern_cells:
                        pattern_neighbors += 1
                
                # Very conservative rule: only fill if clearly justified
                should_fill = False
                
                if below_center and has_reflection:
                    should_fill = True
                elif pattern_neighbors >= 3:  # Completely surrounded
                    should_fill = True
                # Add specific exceptions based on observations
                elif below_center and pattern_neighbors >= 1:
                    # Check if there's a "column" of pattern above
                    has_pattern_above = any((check_r, c) in pattern_cells 
                                          for check_r in range(min_r, r))
                    if has_pattern_above:
                        should_fill = True
                
                if should_fill:
                    result[r][c] = 3
    
    return result

def test_minimal():
    """Test the minimal solution"""
    with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
        task = json.load(f)
    
    all_correct = True
    for i, pair in enumerate(task['train']):
        print(f"\n=== TRAIN PAIR {i+1} ===")
        
        predicted = solve_minimal(pair['input'])
        expected = pair['output']
        
        matches = True
        mismatches = []
        for r in range(len(expected)):
            for c in range(len(expected[0])):
                if predicted[r][c] != expected[r][c]:
                    matches = False
                    mismatches.append((r, c, predicted[r][c], expected[r][c]))
        
        if matches:
            print("✓ PERFECT!")
        else:
            print(f"✗ {len(mismatches)} mismatches")
            # Show all mismatches for debugging
            for r, c, pred, exp in mismatches:
                print(f"  ({r}, {c}): pred={pred}, exp={exp}")
            all_correct = False
    
    return all_correct

# Also try to understand what the EXACT rule should be by working backwards
def reverse_engineer():
    """Work backwards from the correct answers to find the rule"""
    with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
        task = json.load(f)
    
    for i, pair in enumerate(task['train']):
        if i == 2:  # Skip pair 3 (no changes)
            continue
            
        print(f"\n=== REVERSE ENGINEERING PAIR {i+1} ===")
        
        input_arr = np.array(pair['input'])
        output_arr = np.array(pair['output'])
        h, w = input_arr.shape
        
        # Find background
        unique, counts = np.unique(input_arr, return_counts=True)
        bg_color = unique[np.argmax(counts)]
        
        # Find pattern cells
        pattern_cells = set()
        for r in range(h):
            for c in range(w):
                if input_arr[r, c] != bg_color:
                    pattern_cells.add((r, c))
        
        # Find what changed
        changed_cells = set()
        unchanged_bg_cells = set()
        
        for r in range(h):
            for c in range(w):
                if input_arr[r, c] == bg_color:  # Was background
                    if output_arr[r, c] == 3:  # Changed to 3
                        changed_cells.add((r, c))
                    else:  # Stayed background
                        unchanged_bg_cells.add((r, c))
        
        print(f"Changed cells: {len(changed_cells)}")
        print(f"Unchanged background cells: {len(unchanged_bg_cells)}")
        
        # Find what distinguishes changed cells from unchanged ones
        print("\nAnalyzing what makes cells change...")
        
        # Pattern bounding box
        min_r = min(r for r, c in pattern_cells)
        max_r = max(r for r, c in pattern_cells)
        min_c = min(c for r, c in pattern_cells)
        max_c = max(c for r, c in pattern_cells)
        center_r = (min_r + max_r) / 2.0
        
        print(f"Pattern bbox: [{min_r}, {max_r}] x [{min_c}, {max_c}]")
        print(f"Center row: {center_r}")
        
        # Check properties of changed vs unchanged cells
        changed_properties = []
        unchanged_properties = []
        
        for cell_set, prop_list in [(changed_cells, changed_properties), 
                                   (unchanged_bg_cells, unchanged_properties)]:
            for r, c in cell_set:
                if min_r <= r <= max_r and min_c <= c <= max_c:  # In bbox
                    # Properties to check
                    reflected_r = 2 * center_r - r
                    reflected_r_int = round(reflected_r)
                    has_reflection = (min_r <= reflected_r_int <= max_r and 
                                    (reflected_r_int, c) in pattern_cells)
                    
                    below_center = r > center_r
                    above_center = r < center_r
                    
                    pattern_neighbors = sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                                          if 0 <= r+dr < h and 0 <= c+dc < w and 
                                          (r+dr, c+dc) in pattern_cells)
                    
                    prop_list.append({
                        'pos': (r, c),
                        'has_reflection': has_reflection,
                        'below_center': below_center,
                        'above_center': above_center,
                        'pattern_neighbors': pattern_neighbors,
                        'row_distance_from_center': abs(r - center_r)
                    })
        
        # Find distinguishing features
        print(f"\nChanged cells in bbox: {len(changed_properties)}")
        print(f"Unchanged bg cells in bbox: {len(unchanged_properties)}")
        
        # Look for simple rules
        changed_with_reflection = sum(1 for p in changed_properties if p['has_reflection'])
        unchanged_with_reflection = sum(1 for p in unchanged_properties if p['has_reflection'])
        
        print(f"Changed cells with reflection: {changed_with_reflection}/{len(changed_properties)}")
        print(f"Unchanged cells with reflection: {unchanged_with_reflection}/{len(unchanged_properties)}")

if __name__ == '__main__':
    print("Testing minimal approach:")
    success = test_minimal()
    
    print("\n" + "="*50)
    print("REVERSE ENGINEERING:")
    reverse_engineer()
    
    print(f"\nMinimal approach success: {success}")