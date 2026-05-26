#!/usr/bin/env python3

import json
import numpy as np

def verify_hypothesis():
    with open('/Users/evanpieser/apr12_tasks/651b2ee5.json', 'r') as f:
        task = json.load(f)
    
    # Check train pair 1 (9x17) - smallest one to verify manually
    pair = task['train'][0]
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("=== MANUAL VERIFICATION OF TRAIN PAIR 1 ===")
    print("Checking every single cell...")
    
    h, w = output_grid.shape
    print(f"Grid size: {h}x{w}")
    
    # My hypothesis: diagonal X pattern with 8-unit period
    # Let's check if (y+x) % 8 == 0 or (y-x) % 8 == 0 determines marker placement
    
    errors = []
    
    for y in range(h):
        for x in range(w):
            actual = output_grid[y, x]
            
            # Check diagonal patterns
            # Main diagonal: y-x = 0, 8, 16, -8, -16...
            # Anti-diagonal: y+x = 0, 8, 16, 24...
            
            is_diagonal = (y + x) % 8 == 0 or (y - x) % 8 == 0
            
            if is_diagonal:
                expected = 0  # marker color
            else:
                expected = 1  # background color
            
            if actual != expected:
                errors.append((y, x, actual, expected))
    
    if errors:
        print(f"Found {len(errors)} errors with 8-unit diagonal hypothesis:")
        for y, x, actual, expected in errors[:10]:  # Show first 10
            print(f"  ({y},{x}): actual={actual}, expected={expected}")
    else:
        print("✓ 8-unit diagonal pattern matches perfectly!")
    
    # Try different pattern - maybe it's simpler
    print(f"\nTrying different patterns...")
    
    # Pattern 2: Check (y+x) % 4 and (y-x) % 4
    errors2 = []
    for y in range(h):
        for x in range(w):
            actual = output_grid[y, x]
            
            is_diagonal = (y + x) % 4 == 0 or (y - x) % 4 == 0
            
            if is_diagonal:
                expected = 0  # marker color
            else:
                expected = 1  # background color
            
            if actual != expected:
                errors2.append((y, x, actual, expected))
    
    if len(errors2) == 0:
        print("✓ 4-unit diagonal pattern matches perfectly!")
    else:
        print(f"4-unit diagonal pattern has {len(errors2)} errors")
    
    # Let's look at the X pattern more carefully
    # Maybe it's about distance from specific points
    print(f"\nAnalyzing the actual pattern more carefully...")
    
    # Look at where 0s appear in the output
    zero_positions = []
    for y in range(h):
        for x in range(w):
            if output_grid[y, x] == 0:
                zero_positions.append((y, x))
    
    print("Positions with marker color (0):")
    for i, (y, x) in enumerate(zero_positions):
        if i < 20:  # Show first 20
            print(f"  ({y},{x})")
        elif i == 20:
            print(f"  ... and {len(zero_positions) - 20} more")
            break

if __name__ == "__main__":
    verify_hypothesis()