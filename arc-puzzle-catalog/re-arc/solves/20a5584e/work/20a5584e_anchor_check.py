"""
REVISED APPROACH: Check different anchor positions for pattern replication
"""

import json
import numpy as np

def check_pattern_anchoring():
    with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
        task = json.load(f)
    
    # Check pair 2 specifically
    pair = task['train'][1]  
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("PAIR 2 PATTERN ANCHORING CHECK:")
    
    # Original black pattern positions
    original_blacks = [(6, 5), (6, 6), (8, 5)]
    print("Original black pattern:", original_blacks)
    
    # Blue dots
    blue_dots = [(1, 20), (2, 6), (2, 9), (7, 6), (9, 24), (11, 11)]
    print("Blue dots:", blue_dots)
    
    # New black positions
    new_blacks = [(0, 19), (1, 5), (1, 8), (2, 19), (2, 20), (3, 5), (3, 6), (3, 8), (3, 9), (8, 23), (10, 10), (10, 23), (10, 24), (12, 10), (12, 11)]
    print("New black positions:", new_blacks)
    
    # Try different anchor points for the pattern
    # Pattern shape: (6,5), (6,6), (8,5) 
    # Relative to (6,5): (0,0), (0,1), (2,0)
    # Relative to (6,6): (-0,-1), (0,0), (2,-1)  
    # Relative to (8,5): (-2,0), (-2,1), (0,0)
    
    pattern_relative_to_65 = [(0,0), (0,1), (2,0)]
    pattern_relative_to_66 = [(0,-1), (0,0), (2,-1)]
    pattern_relative_to_85 = [(-2,0), (-2,1), (0,0)]
    
    print("\nTesting different anchor points:")
    
    for i, (blue_r, blue_c) in enumerate(blue_dots):
        print(f"\nBlue dot {i+1}: ({blue_r}, {blue_c})")
        
        # Check if any of the patterns match when anchored at this blue dot
        for anchor_name, pattern in [
            ("anchor_65", pattern_relative_to_65),
            ("anchor_66", pattern_relative_to_66), 
            ("anchor_85", pattern_relative_to_85)
        ]:
            matches = []
            for dr, dc in pattern:
                check_r, check_c = blue_r + dr, blue_c + dc
                if (check_r, check_c) in new_blacks:
                    matches.append((dr, dc))
            
            if matches:
                print(f"  {anchor_name}: matches {len(matches)}/3: {matches}")
    
    # Also check if blue dots themselves get replaced by the pattern
    print("\nChecking if blue dots get replaced...")
    for blue_r, blue_c in blue_dots:
        if output_grid[blue_r, blue_c] != 1:  # If blue dot is no longer there
            print(f"Blue dot at ({blue_r}, {blue_c}) replaced with {output_grid[blue_r, blue_c]}")

if __name__ == "__main__":
    check_pattern_anchoring()