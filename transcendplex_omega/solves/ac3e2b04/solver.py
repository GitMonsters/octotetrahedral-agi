#!/usr/bin/env python3
"""ARC puzzle ac3e2b04 solver.

Cross pattern rule:
1. Find all 3x3 blocks (bordered by 3s with 2 in center)
2. For each block's center column: draw vertical cross (0->1, 2->1 at other blocks' edges)
3. For each block's center row: fill 0s with 1s (full row)
4. For each block's edge rows: fill 0s with 1s (only in other blocks' column ranges)
5. For full 2-lines: fill 0s and edge 2s with 1s (within block column ranges)
6. For rows adjacent to 2-lines: fill 0s with 1s (within block column ranges)
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    """Apply the cross pattern rule."""
    import copy
    result = copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find all 3x3 blocks
    blocks = []
    for r in range(rows - 2):
        for c in range(cols - 2):
            if (grid[r][c] == 3 and grid[r][c+1] == 3 and grid[r][c+2] == 3 and
                grid[r+1][c] == 3 and grid[r+1][c+1] == 2 and grid[r+1][c+2] == 3 and
                grid[r+2][c] == 3 and grid[r+2][c+1] == 3 and grid[r+2][c+2] == 3):
                blocks.append((r, c))
    
    # Identify full 2-lines (rows where all non-zero cells are 2 or 3)
    two_lines_rows = set()
    for r in range(rows):
        all_2_or_3 = all(grid[r][c] in [0, 2, 3] for c in range(cols))
        if all_2_or_3:
            count_2 = sum(1 for c in range(cols) if grid[r][c] == 2)
            if count_2 > cols // 2:  # Mostly 2s
                two_lines_rows.add(r)
    
    # Build block column and row ranges
    block_col_ranges = {}  # Maps center_col -> (start_col, end_col)
    block_row_ranges = {}  # Maps center_row -> (start_row, end_row)
    block_edge_cols = set()
    block_edge_rows = set()
    
    for block_r, block_c in blocks:
        center_c = block_c + 1
        center_r = block_r + 1
        block_col_ranges[center_c] = (block_c, block_c + 2)
        block_row_ranges[center_r] = (block_r, block_r + 2)
        block_edge_cols.add(block_c)
        block_edge_cols.add(block_c + 2)
        block_edge_rows.add(block_r)
        block_edge_rows.add(block_r + 2)
    
    # Draw vertical crosses at block centers
    for block_r, block_c in blocks:
        center_c = block_c + 1
        for r in range(rows):
            if grid[r][center_c] == 0:
                result[r][center_c] = 1
            elif grid[r][center_c] == 2 and r in block_edge_rows:
                # Convert 2s to 1s at edge rows of OTHER blocks
                if not (block_r <= r <= block_r + 2):
                    result[r][center_c] = 1
    
    # Draw horizontal crosses at block center rows (convert 0s to 1s)
    for block_r, block_c in blocks:
        center_r = block_r + 1
        for c in range(cols):
            if grid[center_r][c] == 0:
                result[center_r][c] = 1
    
    # Draw horizontal crosses at block edge rows (convert 0s within OTHER blocks' columns)
    for block_r, block_c in blocks:
        for edge_r in [block_r, block_r + 2]:
            # Convert 0s to 1s within OTHER blocks' column ranges
            for center_c, (start_c, end_c) in block_col_ranges.items():
                # Skip if this is the same block
                if center_c == block_c + 1:
                    continue
                for c in range(start_c, end_c + 1):
                    if grid[edge_r][c] == 0:
                        result[edge_r][c] = 1
    
    # Draw horizontal crosses at full 2-lines
    for two_line_r in two_lines_rows:
        # Find all block columns that intersect with this row
        for center_c, (start_c, end_c) in block_col_ranges.items():
            for c in range(start_c, end_c + 1):
                if grid[two_line_r][c] == 0:
                    result[two_line_r][c] = 1
                elif grid[two_line_r][c] == 2 and (c == start_c or c == end_c):
                    # Convert edge 2s to 1s
                    result[two_line_r][c] = 1
    
    # Draw horizontal crosses at rows adjacent to 2-lines
    for two_line_r in two_lines_rows:
        for adjacent_offset in [-1, 1]:
            adjacent_r = two_line_r + adjacent_offset
            if 0 <= adjacent_r < rows:
                # Only apply to 0-rows
                if all(grid[adjacent_r][c] in [0, 3] for c in range(cols)):
                    # Find all block columns
                    for center_c, (start_c, end_c) in block_col_ranges.items():
                        for c in range(start_c, end_c + 1):
                            if grid[adjacent_r][c] == 0:
                                result[adjacent_r][c] = 1
    
    return result


if __name__ == "__main__":
    import json
    
    # Load the puzzle
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/ac3e2b04.json") as f:
        puzzle = json.load(f)
    
    all_pass = True
    for idx, example in enumerate(puzzle["train"]):
        input_grid = example["input"]
        expected = example["output"]
        result = solve(input_grid)
        
        passed = result == expected
        all_pass = all_pass and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"Training example {idx + 1}: {status}")
        
        if not passed:
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape:      {len(result)}x{len(result[0]) if result else 0}")
            # Show first diff
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  First diff at ({r}, {c}): expected {expected[r][c]}, got {result[r][c]}")
                        break
    
    print()
    if all_pass:
        print("All training examples PASSED! ✓")
    else:
        print("Some examples FAILED!")
