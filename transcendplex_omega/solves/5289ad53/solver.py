#!/usr/bin/env python3
"""
ARC-AGI solver for task 5289ad53.

Transformation rule:
- Count horizontal line segments for each non-background color (always 2 and 3 in this puzzle)
- Output is a 2x3 grid where:
  - Row 0 represents color 3 segments
  - Row 1 represents color 2 segments
- Row 0 is filled left-to-right with: min(c3_count, 3) cells of 3, then remainder with min(leftover, c2_count) cells of 2, then 0s
- Row 1 is filled left-to-right with: [3 if c3_count > 3], then min(remaining, c2_count_after_row0) cells of 2, then 0s
  - Special rule: if row 0 used any 2s AND c2_count <= 3, row 1 can only use 1 cell of 2s
"""

from collections import defaultdict
import json
import sys
import os


def get_segments(grid: list[list[int]]) -> dict[int, list[int]]:
    """Count horizontal line segments for each non-background color."""
    bg = grid[0][0]  # Background is top-left color
    segments_by_color = defaultdict(list)
    
    for row in grid:
        in_segment = False
        segment_val = None
        segment_start = None
        
        for col_idx, val in enumerate(row):
            if val != bg:
                if not in_segment:
                    in_segment = True
                    segment_val = val
                    segment_start = col_idx
                elif val != segment_val:
                    # End previous segment
                    length = col_idx - segment_start
                    segments_by_color[segment_val].append(length)
                    segment_val = val
                    segment_start = col_idx
            else:
                if in_segment:
                    # End segment at background
                    length = col_idx - segment_start
                    segments_by_color[segment_val].append(length)
                    in_segment = False
        
        # Handle segment that extends to end of row
        if in_segment:
            length = len(row) - segment_start
            segments_by_color[segment_val].append(length)
    
    return segments_by_color


def solve(grid: list[list[int]]) -> list[list[int]]:
    """Transform input grid to output grid."""
    seg = get_segments(grid)
    c3_count = len(seg.get(3, []))
    c2_count = len(seg.get(2, []))
    
    # Row 0: fill with 3s first, then 2s for remainder
    row0 = []
    row0.extend([3] * min(c3_count, 3))
    remaining_in_row0 = 3 - len(row0)
    row0_2s = min(remaining_in_row0, c2_count)
    row0.extend([2] * row0_2s)
    row0.extend([0] * (3 - len(row0)))
    
    # Row 1
    row1 = []
    
    # If c3_count > 3, start with 3 (overflow indicator)
    if c3_count > 3:
        row1.append(3)
        remaining_in_row1 = 2
    else:
        remaining_in_row1 = 3
    
    # Remaining 2s after row 0 uses some
    remaining_2s = c2_count - row0_2s
    
    # If row 0 used any 2s AND c2_count <= 3, row 1 can only use 1
    if row0_2s > 0 and c2_count <= 3:
        row1_2s = min(1, remaining_2s)
    else:
        row1_2s = min(remaining_in_row1, remaining_2s)
    
    row1.extend([2] * row1_2s)
    row1.extend([0] * (3 - len(row1)))
    
    return [row0, row1]


if __name__ == "__main__":
    # Load task from file (default to specified path)
    task_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5289ad53.json')
    
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    # Test on all training examples
    all_pass = True
    for idx, example in enumerate(task['train']):
        predicted = solve(example['input'])
        expected = example['output']
        
        if predicted == expected:
            print(f"Training example {idx}: PASS")
        else:
            print(f"Training example {idx}: FAIL")
            print(f"  Expected: {expected}")
            print(f"  Got:      {predicted}")
            all_pass = False
    
    # Summary
    if all_pass:
        print("\n✓ All training examples passed!")
        sys.exit(0)
    else:
        print("\n✗ Some training examples failed")
        sys.exit(1)
