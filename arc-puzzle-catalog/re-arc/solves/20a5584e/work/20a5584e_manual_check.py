"""
MANUAL ANALYSIS - THE CORRECT HYPOTHESIS

From the visualization, I can see the actual pattern:

PAIR 1: Blue background (1) with orange shapes (7)
- Input: One orange L-shape in bottom right
- Output: Multiple copies of the same L-shape scattered around

PAIR 2: Orange background (7) with black shapes (0) and blue dots (1)  
- Input: Some black shapes and isolated blue dots
- Output: Black shapes copied around the blue dot positions

PAIR 3: Orange background (7) with red shapes (2), blue dots (1), pink dots (8)
- Input: Red rectangular region at top, blue and pink dots scattered
- Output: Red shapes replicated around blue and pink dots

PAIR 4: Red background (2) with yellow shapes (4) and blue dots (1)
- Input: Yellow L-shape pattern and blue dots scattered
- Output: Yellow shapes replicated around blue dot positions

HYPOTHESIS:
1. Find non-background colored shapes/regions (not color 1, not background)
2. Find isolated dots of color 1 (blue dots)
3. Replicate the shapes around each blue dot position

Wait, let me check where exactly the original shapes are and where they get replicated...
"""

import json
import numpy as np

def manual_verification():
    with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
        task = json.load(f)
    
    # Check pair 2 specifically - it's the clearest
    pair = task['train'][1]  # Pair 2 (0-indexed)
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("PAIR 2 MANUAL CHECK:")
    print("Background color (most frequent):", 7)
    
    # Find original black shapes (color 0) in input
    black_positions = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r, c] == 0:
                black_positions.append((r, c))
    print("Original black positions:", black_positions)
    
    # Find blue dots (color 1) in input
    blue_positions = []
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            if input_grid[r, c] == 1:
                blue_positions.append((r, c))
    print("Blue dot positions:", blue_positions)
    
    # Find new black positions in output
    new_black_positions = []
    for r in range(len(output_grid)):
        for c in range(len(output_grid[0])):
            if output_grid[r, c] == 0 and input_grid[r, c] != 0:
                new_black_positions.append((r, c))
    print("New black positions:", new_black_positions)
    
    # Check if original pattern is replicated near blue dots
    print("\nChecking if pattern is replicated near blue dots...")
    original_pattern_relative = []
    if black_positions:
        min_r = min(pos[0] for pos in black_positions)
        min_c = min(pos[1] for pos in black_positions)
        for r, c in black_positions:
            original_pattern_relative.append((r - min_r, c - min_c))
    print("Original pattern (relative):", original_pattern_relative)
    
    # Check each blue dot
    for blue_r, blue_c in blue_positions:
        print(f"Around blue dot ({blue_r}, {blue_c}):")
        replicated_pattern = []
        for dr, dc in original_pattern_relative:
            check_r, check_c = blue_r + dr, blue_c + dc
            if 0 <= check_r < len(output_grid) and 0 <= check_c < len(output_grid[0]):
                if output_grid[check_r, check_c] == 0 and input_grid[check_r, check_c] != 0:
                    replicated_pattern.append((dr, dc))
        print(f"  Replicated pattern: {replicated_pattern}")

if __name__ == "__main__":
    manual_verification()