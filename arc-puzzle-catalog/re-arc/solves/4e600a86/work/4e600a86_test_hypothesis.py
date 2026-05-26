#!/usr/bin/env python3
"""
New hypothesis: Fill background areas with color 3 to complete a vertically symmetric version of the pattern
"""
import json
import numpy as np
import copy

def load_task(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def solve_hypothesis(grid):
    """
    Hypothesis: Fill background cells with color 3 to make the pattern vertically symmetric
    about its horizontal center line
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
    
    # Find all pattern cells (non-background)
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
    
    # Calculate vertical center
    center_r = (min_r + max_r) / 2.0
    
    print(f"Pattern bbox: [{min_r}, {max_r}] x [{min_c}, {max_c}]")
    print(f"Center row: {center_r}")
    print(f"Pattern cells: {len(pattern_cells)}")
    
    # For each background cell in the bounding box, check if making it symmetric
    # would require filling it with color 3
    changes = 0
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == bg_color:  # Background cell
                # Calculate its vertical reflection
                reflected_r = 2 * center_r - r
                reflected_r_int = round(reflected_r)
                
                # Check if the reflected position has a pattern cell
                if (min_r <= reflected_r_int <= max_r and 
                    (reflected_r_int, c) in pattern_cells):
                    result[r][c] = 3
                    changes += 1
    
    print(f"Made {changes} changes")
    return result

def test_hypothesis():
    task = load_task('/Users/evanpieser/apr12_tasks/4e600a86.json')
    
    all_correct = True
    for i, pair in enumerate(task['train']):
        print(f"\n=== TESTING TRAIN PAIR {i+1} ===")
        predicted = solve_hypothesis(pair['input'])
        expected = pair['output']
        
        # Check if they match
        matches = True
        for r in range(len(expected)):
            for c in range(len(expected[0])):
                if predicted[r][c] != expected[r][c]:
                    matches = False
                    print(f"MISMATCH at ({r}, {c}): predicted {predicted[r][c]}, expected {expected[r][c]}")
        
        if matches:
            print("✓ CORRECT!")
        else:
            print("✗ INCORRECT!")
            all_correct = False
    
    print(f"\n=== FINAL RESULT ===")
    print(f"All training pairs correct: {all_correct}")
    return all_correct

if __name__ == '__main__':
    test_hypothesis()