#!/usr/bin/env python3
"""
ARC Task 143845c2 Solver

Transform H×W input to 3H×3W output by upscaling with neighborhood encoding.

For each input cell [r,c]:
- Extract its 3×3 neighborhood (padding with background for edges)
- This neighborhood becomes the 3×3 output block for the cell
- The cell's own position in the block corresponds to the cell itself
- Each position encodes its 8-neighbor value in the input
"""

import json
from collections import Counter

def identify_colors(grid):
    """Identify background (most common) and foreground colors, treating 9 as background."""
    flat = [v for row in grid for v in row]
    counts = Counter([v for v in flat if v != 9])
    if not counts:
        return 0, 1
    fg = min(counts, key=counts.get)  # Least common
    bg = max(counts, key=counts.get)  # Most common
    return bg, fg

def solve(input_grid):
    """Transform H×W input to 3H×3W output."""
    H, W = len(input_grid), len(input_grid[0])
    bg, fg = identify_colors(input_grid)
    
    # Normalize input (treat 9 as bg)
    normalized = []
    for row in input_grid:
        norm_row = [bg if v == 9 else v for v in row]
        normalized.append(norm_row)
    
    output = [[bg] * (3 * W) for _ in range(3 * H)]
    
    # For each input cell, extract its 3×3 neighborhood and place it in the output
    for r in range(H):
        for c in range(W):
            # Extract the 3×3 neighborhood centered at [r,c]
            # This becomes the output block for this cell
            for br in range(3):
                for bc in range(3):
                    # Neighborhood row/col
                    nr = r + br - 1
                    nc = c + bc - 1
                    
                    # Get value from neighborhood (pad with bg for out-of-bounds)
                    if 0 <= nr < H and 0 <= nc < W:
                        val = normalized[nr][nc]
                    else:
                        val = bg
                    
                    # Place in output block
                    output[r*3 + br][c*3 + bc] = val
    
    return output

def main():
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
        
        # Test on training examples
        for i, example in enumerate(data.get('train', [])):
            input_grid = example['input']
            expected_output = example['output']
            
            predicted = solve(input_grid)
            
            # Check if prediction matches
            if predicted == expected_output:
                print(f"✓ Training example {i}: CORRECT")
            else:
                print(f"✗ Training example {i}: INCORRECT")
                print(f"  Input: {len(input_grid)}×{len(input_grid[0])}")
                print(f"  Expected output size: {len(expected_output)}×{len(expected_output[0])}")
                print(f"  Got output size: {len(predicted)}×{len(predicted[0])}")
                
                # Count mismatches
                mismatches = sum(1 for r in range(len(predicted)) for c in range(len(predicted[0]))
                                if predicted[r][c] != expected_output[r][c])
                total = len(predicted) * len(predicted[0])
                print(f"  Mismatches: {mismatches}/{total} ({100*mismatches/total:.1f}%)")

if __name__ == '__main__':
    main()



def main():
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
        
        # Test on training examples
        for i, example in enumerate(data.get('train', [])):
            input_grid = example['input']
            expected_output = example['output']
            
            predicted = solve(input_grid)
            
            # Check if prediction matches
            if predicted == expected_output:
                print(f"✓ Training example {i}: CORRECT")
            else:
                print(f"✗ Training example {i}: INCORRECT")
                print(f"  Input: {len(input_grid)}×{len(input_grid[0])}")
                print(f"  Expected output size: {len(expected_output)}×{len(expected_output[0])}")
                print(f"  Got output size: {len(predicted)}×{len(predicted[0])}")
                
                # Count mismatches
                mismatches = sum(1 for r in range(len(predicted)) for c in range(len(predicted[0]))
                                if predicted[r][c] != expected_output[r][c])
                total = len(predicted) * len(predicted[0])
                print(f"  Mismatches: {mismatches}/{total}")

if __name__ == '__main__':
    main()
