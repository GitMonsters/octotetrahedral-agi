#!/usr/bin/env python3
"""
Solver for ARC-AGI task 5833af48.

The task takes two patterns (pattern1 and pattern2) separated by zero columns/rows:
- Pattern1: A small template with color 2 and 8s
- Pattern2: A larger pattern with another color (X) and 8s
- Background: Filled with color X

The output is constructed as:
- Top half: Pattern1's 8 positions shifted to center, rest filled with background color
- Bottom half: Pattern1 repeated at left and right edges, middle filled with background
- Output dimensions: (pattern1_height * 2) x (pattern2_width * 2)
"""

import json
import numpy as np


def solve(grid):
    """
    Solve the ARC puzzle for input grid.
    
    Args:
        grid: List of lists representing the input grid
        
    Returns:
        List of lists representing the output grid, or None if parsing fails
    """
    inp = np.array(grid)
    
    # Find row separator (first all-zero row after row 0)
    sep_row = None
    for r in range(inp.shape[0]):
        if r > 0 and np.all(inp[r] == 0):
            sep_row = r
            break
    
    if sep_row is None:
        return None
    
    # Extract top part (between row 0 and sep_row)
    top_part = inp[1:sep_row]
    
    # Find non-zero columns
    nonzero_cols = [c for c in range(top_part.shape[1]) if np.any(top_part[:, c] != 0)]
    
    if not nonzero_cols:
        return None
    
    # Find column gap (separator between pattern1 and pattern2)
    col_gap = None
    for i in range(len(nonzero_cols) - 1):
        if nonzero_cols[i + 1] - nonzero_cols[i] > 1:
            col_gap = (nonzero_cols[i], nonzero_cols[i + 1])
            break
    
    if col_gap is None:
        return None
    
    # Extract pattern regions
    p1_cols = [c for c in nonzero_cols if c <= col_gap[0]]
    p2_cols = [c for c in nonzero_cols if c >= col_gap[1]]
    
    if not p1_cols or not p2_cols:
        return None
    
    p1_start, p1_end = min(p1_cols), max(p1_cols)
    p2_start, p2_end = min(p2_cols), max(p2_cols)
    
    p1 = inp[1:sep_row, p1_start:p1_end + 1]
    p2 = inp[1:sep_row, p2_start:p2_end + 1]
    
    # Get background color from bottom section
    bottom = inp[sep_row + 1:, p2_start:p2_end + 1]
    bg_color = None
    for val in np.unique(bottom):
        if val > 0:
            bg_color = int(val)
            break
    
    if bg_color is None:
        return None
    
    # Calculate bottom region height
    bottom_rows = [r for r in range(sep_row + 1, inp.shape[0]) if np.any(inp[r] != 0)]
    bottom_height = max(bottom_rows) - min(bottom_rows) + 1 if bottom_rows else 0
    
    if bottom_height == 0:
        return None
    
    # Count actual content rows in p1 (exclude all-zero rows)
    p1_h_full, p1_w = p1.shape
    p1_h = 0
    for r in range(p1_h_full):
        if np.any(p1[r] != 0):
            p1_h += 1
        else:
            break  # Stop at first all-zero row
    p2_h, p2_w = p2.shape
    
    # Output dimensions: height = bottom_height
    # Scale is based on the content height of p1
    scale = bottom_height // p1_h if p1_h > 0 else 1
    if bottom_height % p1_h != 0:
        scale += 1
    
    out_h = bottom_height
    
    # Width calculation based on scale and p1 structure
    # If p1 has all rows with content (p1_h_full == p1_h), use p2_w * scale
    # Otherwise (p1 has empty rows), use p1_w + p2_w
    if p1_h_full == p1_h:
        # All rows have content
        if scale == 1:
            out_w = p1_w + p2_w
        else:
            out_w = p2_w * scale
    else:
        # Some rows are empty, use combined width
        out_w = p1_w + p2_w
    
    result = np.full((out_h, out_w), bg_color, dtype=inp.dtype)
    
    # The pattern repeats scale times vertically
    # On even repeats (0, 2, 4, ...), place pattern1's 8s at a center offset
    # On odd repeats (1, 3, 5, ...), place pattern1 on left and right edges
    
    # Center offset determination:
    # The condition checks if we're in a "scaled width" scenario (p2_w * scale)
    # versus a "combined width" scenario (p1_w + p2_w)
    if out_w == p2_w * scale:
        # Scaled width scenario
        if scale > 1 and p2_w > 0 and p1_w == p2_w // 2:
            center_offset = (out_w - p2_w) // 2
        else:
            center_offset = (out_w - p1_w) // 2
    else:
        # Combined width scenario (p1_w + p2_w)
        center_offset = (out_w - p1_w) // 2
    
    for tile_idx in range(scale):
        row_base = tile_idx * p1_h
        
        if tile_idx % 2 == 0:  # Even repeats: center pattern
            for r in range(min(p1_h, out_h - row_base)):
                for c in range(p1_w):
                    if p1[r, c] == 8:
                        out_col = c + center_offset
                        if 0 <= out_col < out_w:
                            result[row_base + r, out_col] = 8
        else:  # Odd repeats: left and right edges
            for r in range(min(p1_h, out_h - row_base)):
                for c in range(p1_w):
                    if p1[r, c] == 8:
                        # Left edge
                        result[row_base + r, c] = 8
                        # Right edge
                        result[row_base + r, out_w - p1_w + c] = 8
    
    return result.tolist()


if __name__ == "__main__":
    # Load task JSON
    with open('~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5833af48.json'.replace('~', '/Users/evanpieser'), 'r') as f:
        task = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for i, example in enumerate(task['train']):
        computed = solve(example['input'])
        expected = example['output']
        
        if computed is None:
            print(f"Training example {i + 1}: FAIL (returned None)")
            all_pass = False
        else:
            match = np.array_equal(np.array(computed), np.array(expected))
            status = "PASS" if match else "FAIL"
            print(f"Training example {i + 1}: {status}")
            if not match:
                all_pass = False
    
    # Summary
    print(f"\n{'All training examples PASSED!' if all_pass else 'Some training examples FAILED!'}")
