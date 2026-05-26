"""
Debug pair 2 specifically to understand what's wrong
"""

import json
import numpy as np

def debug_pair_2():
    with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
        task = json.load(f)
    
    pair = task['train'][1]
    input_grid = np.array(pair['input'])
    expected_output = np.array(pair['output'])
    
    print("PAIR 2 DEBUG:")
    print(f"Grid size: {input_grid.shape}")
    
    # Background is 7
    background = 7
    
    # Blue dots
    blue_dots = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r, c] == 1:
                blue_dots.append((r, c))
    print(f"Blue dots: {blue_dots}")
    
    # Check each blue dot and what should happen around it
    pattern_offsets = [(-1, -1), (1, -1), (1, 0)]
    
    for blue_r, blue_c in blue_dots:
        print(f"\nBlue dot at ({blue_r}, {blue_c}):")
        for dr, dc in pattern_offsets:
            target_r, target_c = blue_r + dr, blue_c + dc
            if 0 <= target_r < len(input_grid) and 0 <= target_c < len(input_grid[0]):
                input_val = input_grid[target_r, target_c]
                expected_val = expected_output[target_r, target_c]
                print(f"  Offset ({dr},{dc}) -> ({target_r},{target_c}): input={input_val}, expected={expected_val}")
                
                # Check if there's an existing pattern at this blue dot location
                if input_val != background and input_val != 1:
                    print(f"    *** Conflict: position has existing pattern color {input_val}")
    
    # Check if there are blue dots that DON'T get patterns
    blue_dots_without_pattern = []
    for blue_r, blue_c in blue_dots:
        has_pattern = False
        for dr, dc in pattern_offsets:
            target_r, target_c = blue_r + dr, blue_c + dc
            if 0 <= target_r < len(input_grid) and 0 <= target_c < len(input_grid[0]):
                if expected_output[target_r, target_c] != input_grid[target_r, target_c]:
                    has_pattern = True
                    break
        if not has_pattern:
            blue_dots_without_pattern.append((blue_r, blue_c))
    
    print(f"\nBlue dots without pattern replication: {blue_dots_without_pattern}")

if __name__ == "__main__":
    debug_pair_2()