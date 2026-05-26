"""
Let me look at this more systematically by checking exact coordinates
"""

import json
import numpy as np

def systematic_check():
    with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
        task = json.load(f)
    
    # Pair 2: Most clear example
    pair = task['train'][1]  
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("PAIR 2 SYSTEMATIC CHECK:")
    print("Grid dimensions:", input_grid.shape)
    
    # Find all blue dots (1s) and their neighborhoods
    blue_dots = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r, c] == 1:
                blue_dots.append((r, c))
    
    print("Blue dots:", blue_dots)
    
    # For each blue dot, check what appears in a 5x5 neighborhood in output vs input
    for blue_r, blue_c in blue_dots:
        print(f"\nBlue dot at ({blue_r}, {blue_c}):")
        neighborhood_changes = []
        
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = blue_r + dr, blue_c + dc
                if 0 <= nr < len(input_grid) and 0 <= nc < len(input_grid[0]):
                    if input_grid[nr, nc] != output_grid[nr, nc]:
                        neighborhood_changes.append((dr, dc, input_grid[nr, nc], output_grid[nr, nc]))
        
        if neighborhood_changes:
            print(f"  Changes: {neighborhood_changes}")
        else:
            print("  No changes in neighborhood")

if __name__ == "__main__":
    systematic_check()