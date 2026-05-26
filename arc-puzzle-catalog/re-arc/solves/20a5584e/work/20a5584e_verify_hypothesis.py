"""
FINAL HYPOTHESIS VERIFICATION

The pattern is: 
- Original black shape at (6,5), (6,6), (8,5) 
- Relative to (7,6): [(-1,-1), (-1,0), (1,-1)]
- But we're seeing [(-1,-1), (1,-1), (1,0)] around blue dots

Let me check the original pattern more carefully...
"""

import json
import numpy as np

def verify_hypothesis():
    with open('/Users/evanpieser/apr12_tasks/20a5584e.json', 'r') as f:
        task = json.load(f)
    
    # Pair 2
    pair = task['train'][1]  
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    
    print("PAIR 2 HYPOTHESIS VERIFICATION:")
    
    # Original black positions: (6,5), (6,6), (8,5)
    original_blacks = [(6,5), (6,6), (8,5)]
    print("Original black positions:", original_blacks)
    
    # Blue dots that get pattern replication
    blue_dots = [(1, 20), (2, 6), (2, 9), (9, 24), (11, 11)]
    
    # Expected pattern around each blue dot: [(-1,-1), (1,-1), (1,0)]
    expected_pattern = [(-1, -1), (1, -1), (1, 0)]
    
    print("Expected pattern relative to blue dot:", expected_pattern)
    print("\nVerifying pattern replication:")
    
    for blue_r, blue_c in blue_dots:
        print(f"\nBlue dot at ({blue_r}, {blue_c}):")
        matches = 0
        for dr, dc in expected_pattern:
            check_r, check_c = blue_r + dr, blue_c + dc
            if 0 <= check_r < len(output_grid) and 0 <= check_c < len(output_grid[0]):
                if output_grid[check_r, check_c] == 0 and input_grid[check_r, check_c] == 7:
                    matches += 1
                    print(f"  ✓ Pattern at ({check_r}, {check_c}) = {dr}, {dc}")
                else:
                    print(f"  ✗ No pattern at ({check_r}, {check_c}) = {dr}, {dc}")
        print(f"  Matches: {matches}/3")
    
    # Now let's check if this hypothesis works for other pairs
    print("\n" + "="*50)
    print("CHECKING OTHER PAIRS:")
    
    for pair_idx in [0, 2, 3]:  # Skip pair 1 for now
        print(f"\n--- PAIR {pair_idx + 1} ---")
        pair = task['train'][pair_idx]
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        # Find background color
        from collections import Counter
        color_counts = Counter(input_grid.flatten())
        background = max(color_counts, key=color_counts.get)
        print(f"Background color: {background}")
        
        # Find blue dots (1s)
        blue_dots = []
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                if input_grid[r, c] == 1:
                    blue_dots.append((r, c))
        print(f"Blue dots: {blue_dots}")
        
        # Find original pattern (non-background, non-1 colors)
        pattern_colors = set()
        for r in range(len(input_grid)):
            for c in range(len(input_grid[0])):
                if input_grid[r, c] != background and input_grid[r, c] != 1:
                    pattern_colors.add(input_grid[r, c])
        print(f"Pattern colors: {pattern_colors}")
        
        # Find original pattern positions
        if pattern_colors:
            original_pattern_positions = []
            for color in pattern_colors:
                for r in range(len(input_grid)):
                    for c in range(len(input_grid[0])):
                        if input_grid[r, c] == color:
                            original_pattern_positions.append((r, c, color))
            print(f"Original pattern positions: {original_pattern_positions}")

if __name__ == "__main__":
    verify_hypothesis()