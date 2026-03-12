#!/usr/bin/env python3
import json
import sys
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Extract base repeating pattern from a complete row, then fill zeros.
    Each row is an offset version of the same base pattern that shifts by 1 per row.
    """
    grid = [row[:] for row in grid]  # Deep copy
    h, w = len(grid), len(grid[0])
    
    # Find a complete row (no zeros) to extract the pattern
    complete_rows = []
    for row_idx in range(3, h):
        row = grid[row_idx]
        if 0 not in row[3:] and not all(x == 4 for x in row):
            complete_rows.append(row_idx)
    
    if not complete_rows:
        return grid
    
    ref_row_idx = complete_rows[0]
    ref_row = grid[ref_row_idx]
    
    # Extract the repeating pattern from the complete row
    pattern = _extract_pattern(ref_row[3:])
    
    if not pattern:
        return grid
    
    # Find the offset of the reference row
    # (this is determined by what value appears at column 3)
    ref_offset = pattern.index(ref_row[3]) if ref_row[3] in pattern else 0
    
    # Fill all rows
    for row_idx in range(3, h):
        row = grid[row_idx]
        
        # Skip border rows
        if all(x == 4 for x in row) or all(x == 1 for x in row):
            continue
        
        # Calculate offset for this row: increments by (row_idx - ref_row_idx) from ref
        row_offset = (ref_offset + (row_idx - ref_row_idx)) % len(pattern)
        
        # Fill zeros using the pattern at this offset
        for col_idx in range(3, w):
            if row[col_idx] == 0:
                pattern_idx = (col_idx - 3 + row_offset) % len(pattern)
                row[col_idx] = pattern[pattern_idx]
    
    return grid


def _extract_pattern(seq: List[int]) -> List[int]:
    """Find the minimal repeating unit in a sequence."""
    for period in range(1, len(seq) // 2 + 1):
        is_valid = True
        for i in range(len(seq) - period):
            if seq[i] != seq[i + period]:
                is_valid = False
                break
        if is_valid:
            return seq[:period]
    return seq


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
    import os
    sys.exit(main())



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
    import os
    sys.exit(main())



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
    import os
    sys.exit(main())
