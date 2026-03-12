#!/usr/bin/env python3
import json
import sys
import os
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Fill all 0s using repeating row patterns.
    Data rows always start with [1, 1, 4].
    Columns 3+ follow a repeating pattern specific to each row.
    """
    grid = [row[:] for row in grid]  # Deep copy
    h, w = len(grid), len(grid[0])
    
    # Find a reference row with a clear repeating pattern in columns 3+
    pattern = None
    ref_row_idx = None
    
    for row_idx in range(3, h):
        row = grid[row_idx]
        if 0 in row[3:]:
            continue
        if all(x == 4 for x in row):
            continue
        
        candidate_pattern = _extract_pattern(row[3:])
        if candidate_pattern and 1 < len(candidate_pattern) <= 10:
            pattern = candidate_pattern
            ref_row_idx = row_idx
            break
    
    if pattern is None:
        return grid
    
    ref_offset = pattern.index(grid[ref_row_idx][3]) if grid[ref_row_idx][3] in pattern else 0
    
    # Fill all zeros
    for row_idx in range(h):
        row = grid[row_idx]
        
        if not any(x == 0 for x in row):
            continue
        
        # Determine row type based on non-zero values
        non_zero_vals = set(x for x in row if x != 0)
        
        # Pure border rows (only 1s)
        if non_zero_vals == {1}:
            for col_idx in range(len(row)):
                if row[col_idx] == 0:
                    row[col_idx] = 1
        # Separator rows (contains both 1 and 4)
        elif non_zero_vals == {1, 4} and any(row[i] == 4 for i in range(3, len(row))):
            for col_idx in range(len(row)):
                if row[col_idx] == 0:
                    row[col_idx] = 4 if col_idx >= 3 else 1
        # Data rows (contains values besides 1 and 4)
        else:
            # Always start with [1, 1, 4]
            for col_idx in range(3):
                if row[col_idx] == 0:
                    row[col_idx] = [1, 1, 4][col_idx]
            
            # Fill columns 3+ with repeating pattern
            row_offset = (ref_offset + (row_idx - ref_row_idx)) % len(pattern)
            for col_idx in range(3, w):
                if row[col_idx] == 0:
                    pattern_idx = (col_idx - 3 + row_offset) % len(pattern)
                    row[col_idx] = pattern[pattern_idx]
    
    return grid


def _extract_pattern(seq: List[int]) -> List[int]:
    """Find the minimal repeating unit in a sequence."""
    for period in range(1, len(seq) // 2 + 1):
        if all(seq[i] == seq[i % period] for i in range(len(seq))):
            return seq[:period]
    return None


def main():
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/4aab4007.json"
    
    json_path = json_path.replace("~", os.path.expanduser("~"))
    
    with open(json_path) as f:
        data = json.load(f)
    
    all_pass = True
    for i, example in enumerate(data["train"]):
        result = solve(example["input"])
        expected = example["output"]
        
        if result == expected:
            print(f"Training example {i}: PASS")
        else:
            print(f"Training example {i}: FAIL")
            all_pass = False
    
    if all_pass:
        print("\nAll training examples PASSED!")
    else:
        print("\nSome training examples FAILED!")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
