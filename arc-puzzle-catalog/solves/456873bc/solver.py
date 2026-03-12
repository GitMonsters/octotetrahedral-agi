import json
import sys
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    ARC-AGI task 456873bc solver.
    
    The transformation:
    1. Identifies a "mask" region filled with 3s
    2. Extends the repeating 4x4 pattern to fill the masked area
    3. Marks specific 2s as 8s based on their position in the repeating pattern
    """
    rows, cols = len(grid), len(grid[0])
    mask_rows = set()
    mask_cols = set()
    
    # Find mask region
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 3:
                mask_rows.add(i)
                mask_cols.add(j)
    
    # Create output with mask removed (3 -> 0)
    output = []
    for i in range(rows):
        row = []
        for j in range(cols):
            val = grid[i][j]
            row.append(0 if val == 3 else val)
        output.append(row)
    
    if not mask_rows or not mask_cols:
        return output
    
    mask_r_min, mask_r_max = min(mask_rows), max(mask_rows)
    mask_c_min, mask_c_max = min(mask_cols), max(mask_cols)
    
    # Find the base 4x4 repeating pattern
    base_pattern = None
    for start_i in range(rows - 3):
        for start_j in range(cols - 3):
            block_valid = True
            for bi in range(4):
                for bj in range(4):
                    if (start_i + bi) in mask_rows or (start_j + bj) in mask_cols:
                        block_valid = False
                        break
                if not block_valid:
                    break
            
            if block_valid:
                base_pattern = []
                for bi in range(4):
                    base_pattern.append(output[start_i + bi][start_j:start_j + 4])
                break
        if base_pattern:
            break
    
    if not base_pattern:
        return output
    
    # Fill mask region with extended pattern
    for i in range(mask_r_min, mask_r_max + 1):
        for j in range(mask_c_min, mask_c_max + 1):
            output[i][j] = base_pattern[i % 4][j % 4]
    
    # Mark specific 2s as 8s
    # Rule: 2->8 if this pattern position (pi, pj) has at least one occurrence become 8
    # Based on analysis: certain positions in repeating pattern are marked as 8
    # We'll use heuristic: mark 2 at (i, j) as 8 if:
    # - It's a 2 in the original grid AND
    # - Its position aligns with mask boundaries in a specific way
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 2:
                pi, pj = i % 4, j % 4
                
                # Check if this specific (i, j) should become 8
                # Based on empirical analysis: 2->8 when pattern position appears
                # at locations near or reaching into mask
                
                # Check which instances of pattern (pi, pj) should be marked
                is_at_mask_boundary_row = (i == mask_r_min or i == mask_r_max)
                is_at_mask_boundary_col = (j == mask_c_min or j == mask_c_max)
                is_in_mask_row_range = (mask_r_min <= i <= mask_r_max)
                is_in_mask_col_range = (mask_c_min <= j <= mask_c_max)
                
                # Heuristic: mark as 8 if it's at a "corner" of visible pattern blocks
                # that would extend toward the mask
                should_mark_8 = False
                
                if is_in_mask_row_range and is_in_mask_col_range:
                    should_mark_8 = True
                elif is_in_mask_row_range and j == max(j for jj in range(cols) if grid[i][jj] != 3):
                    should_mark_8 = (pj == (max(jj for jj in range(cols) if grid[i][jj] != 3) % 4))
                elif is_in_mask_col_range and i == max(i for ii in range(rows) if grid[ii][j] != 3):
                    should_mark_8 = (pi == (max(ii for ii in range(rows) if grid[ii][j] != 3) % 4))
                
                if should_mark_8 and output[i][j] == 2:
                    output[i][j] = 8
    
    return output


def main():
    # Load task JSON
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/456873bc.json")
    
    with open(json_path) as f:
        task_data = json.load(f)
    
    # Test on training examples
    all_pass = True
    for idx, example in enumerate(task_data["train"]):
        result = solve(example["input"])
        expected = example["output"]
        
        # Check if match
        matches = result == expected
        if matches:
            print(f"PASS: Training example {idx}")
        else:
            print(f"FAIL: Training example {idx}")
            all_pass = False
            # Print first difference
            for i in range(len(result)):
                for j in range(len(result[0])):
                    if result[i][j] != expected[i][j]:
                        print(f"  First diff at ({i},{j}): got {result[i][j]}, expected {expected[i][j]}")
                        break
                else:
                    continue
                break
    
    if all_pass:
        print("\nAll training examples PASS!")
    else:
        print("\nSome training examples FAILED!")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    import os
    sys.exit(main())
