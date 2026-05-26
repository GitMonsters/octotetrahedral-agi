#!/usr/bin/env python3
"""
New approach: Complete the pattern to make it symmetric by filling holes/gaps
"""
import json
import numpy as np
import copy

def solve_v2(grid):
    """
    Fill background cells with 3 where doing so would complete a symmetric pattern
    """
    result = copy.deepcopy(grid)
    h, w = len(grid), len(grid[0])
    
    # Find background color
    colors = {}
    for row in grid:
        for cell in row:
            colors[cell] = colors.get(cell, 0) + 1
    bg_color = max(colors, key=colors.get)
    
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
    
    # Try different approach: For each background cell, check if it should be filled
    # based on the pattern around it
    
    # Strategy: Look for "enclosed" background areas or areas that would complete
    # obvious geometric patterns
    
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == bg_color:
                # Check various conditions for when to fill this cell
                
                # Condition 1: Vertical reflection (but with some constraints)
                center_r = (min_r + max_r) / 2.0
                reflected_r = 2 * center_r - r
                reflected_r_int = round(reflected_r)
                
                has_vertical_reflection = (min_r <= reflected_r_int <= max_r and 
                                         (reflected_r_int, c) in pattern_cells)
                
                # Condition 2: Check if filling would create better symmetry
                # (Count pattern neighbors)
                pattern_neighbors = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in pattern_cells:
                        pattern_neighbors += 1
                
                # Heuristic: Fill if has vertical reflection AND some pattern neighbors
                # OR if it has many pattern neighbors (enclosed area)
                should_fill = (has_vertical_reflection and pattern_neighbors >= 1) or pattern_neighbors >= 3
                
                if should_fill:
                    result[r][c] = 3
    
    return result

def test_solve():
    """Test the new solution"""
    task_file = '/Users/evanpieser/apr12_tasks/4e600a86.json'
    with open(task_file, 'r') as f:
        task = json.load(f)
    
    all_correct = True
    for i, pair in enumerate(task['train']):
        print(f"\n=== TRAIN PAIR {i+1} ===")
        predicted = solve_v2(pair['input'])
        expected = pair['output']
        
        matches = True
        mismatches = []
        for r in range(len(expected)):
            for c in range(len(expected[0])):
                if predicted[r][c] != expected[r][c]:
                    matches = False
                    mismatches.append((r, c, predicted[r][c], expected[r][c]))
        
        if matches:
            print("✓ CORRECT!")
        else:
            print(f"✗ {len(mismatches)} mismatches:")
            for r, c, pred, exp in mismatches[:5]:  # Show first 5
                print(f"  ({r}, {c}): predicted {pred}, expected {exp}")
            if len(mismatches) > 5:
                print(f"  ... and {len(mismatches) - 5} more")
            all_correct = False
    
    return all_correct

if __name__ == '__main__':
    success = test_solve()
    print(f"\nOverall success: {success}")