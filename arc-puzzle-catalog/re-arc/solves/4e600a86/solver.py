#!/usr/bin/env python3
"""
Final solution attempt: Fill background areas to complete the shape
"""
import json
import numpy as np
import copy

def solve(grid):
    """
    Transform the grid by filling background areas with color 3 based on pattern completion
    """
    result = copy.deepcopy(grid)
    
    if not grid or not grid[0]:
        return result
    
    h, w = len(grid), len(grid[0])
    
    # Find background color (most frequent)
    colors = {}
    for row in grid:
        for cell in row:
            colors[cell] = colors.get(cell, 0) + 1
    bg_color = max(colors, key=colors.get)
    
    # If background is already 3, no change needed
    if bg_color == 3:
        return result
    
    # Find all pattern cells
    pattern_cells = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg_color:
                pattern_cells.add((r, c))
    
    if not pattern_cells:
        return result
    
    # Find pattern bounding box
    min_r = min(r for r, c in pattern_cells)
    max_r = max(r for r, c in pattern_cells)
    min_c = min(c for r, c in pattern_cells)
    max_c = max(c for r, c in pattern_cells)
    
    center_r = (min_r + max_r) / 2.0
    
    # Strategy: Fill background cells that meet certain criteria
    # Based on the analysis, it seems like we need to fill areas that would
    # "complete" the pattern in some geometric sense
    
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == bg_color:  # Background cell in bounding box
                
                # Criteria for filling:
                should_fill = False
                
                # 1. Has vertical reflection in pattern
                reflected_r = 2 * center_r - r
                reflected_r_int = round(reflected_r)
                has_vertical_reflection = (min_r <= reflected_r_int <= max_r and 
                                         (reflected_r_int, c) in pattern_cells)
                
                # 2. Count pattern neighbors
                pattern_neighbors = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in pattern_cells:
                        pattern_neighbors += 1
                
                # 3. Check if it's in a "filled" region (surrounded by pattern or filled cells)
                # Count pattern + already filled neighbors
                filled_neighbors = pattern_neighbors
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == 3:
                        filled_neighbors += 1
                
                # Heuristic decision rules based on the observed patterns:
                
                # Rule 1: Fill if has vertical reflection (most common case)
                if has_vertical_reflection:
                    should_fill = True
                
                # Rule 2: Fill if it has 2+ pattern neighbors (enclosed area)
                elif pattern_neighbors >= 2:
                    should_fill = True
                
                # Rule 3: Fill specific positions that seem to complete the shape
                # (This is based on the exceptions we saw)
                elif r > center_r:  # Below center
                    # Check if there's a pattern cell above at same column
                    for check_r in range(min_r, r):
                        if (check_r, c) in pattern_cells:
                            should_fill = True
                            break
                
                if should_fill:
                    result[r][c] = 3
    
    return result

def test_solution():
    """Test the solution on all training pairs"""
    with open('/Users/evanpieser/apr12_tasks/4e600a86.json', 'r') as f:
        task = json.load(f)
    
    all_correct = True
    for i, pair in enumerate(task['train']):
        print(f"\n=== TESTING TRAIN PAIR {i+1} ===")
        
        predicted = solve(pair['input'])
        expected = pair['output']
        
        # Check if they match exactly
        matches = True
        mismatches = []
        for r in range(len(expected)):
            for c in range(len(expected[0])):
                if predicted[r][c] != expected[r][c]:
                    matches = False
                    mismatches.append((r, c, predicted[r][c], expected[r][c]))
        
        if matches:
            print("✓ PERFECT MATCH!")
        else:
            print(f"✗ {len(mismatches)} mismatches")
            for r, c, pred, exp in mismatches[:10]:  # Show first 10
                print(f"  ({r}, {c}): predicted {pred}, expected {exp}")
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more")
            all_correct = False
    
    print(f"\n{'='*50}")
    print(f"OVERALL RESULT: {'SUCCESS' if all_correct else 'FAILED'}")
    print(f"{'='*50}")
    
    return all_correct

if __name__ == '__main__':
    success = test_solution()
    
    if success:
        print("\n🎉 All training pairs pass! Saving solution...")
        # The solve function is already defined above
    else:
        print("\n❌ Solution needs more work...")