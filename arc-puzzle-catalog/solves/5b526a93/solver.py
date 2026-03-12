import json
from typing import List, Set

def solve(grid: List[List[int]]) -> List[List[int]]:
    """
    Rule:
    1. Find all column positions where [1,1,1] appears in any row (3-wide blocks of all 1s)
    2. These are the 3x3 block positions
    3. For each row group, fill all-zero blocks at these positions using 8s
    """
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0]) if grid else 0
    
    if rows < 3 or cols < 3:
        return result
    
    # Find all column positions with [1, 1, 1]
    block_positions: Set[int] = set()
    
    for row in grid:
        for col_start in range(len(row) - 2):
            if row[col_start] == 1 and row[col_start + 1] == 1 and row[col_start + 2] == 1:
                block_positions.add(col_start)
    
    block_positions = sorted(block_positions)
    
    if not block_positions:
        return result
    
    # Process each row group
    for row_start in range(0, rows, 3):
        row_end = min(row_start + 3, rows)
        if row_end - row_start < 3:
            continue
        
        # Find template from first block with 1s
        template_shape = None
        template_col = None
        
        for block_col in block_positions:
            if block_col + 3 <= cols:
                block = [grid[r][block_col:block_col+3] for r in range(row_start, row_end)]
                has_ones = any(v == 1 for row in block for v in row)
                
                if has_ones and template_shape is None:
                    template_shape = tuple(tuple(1 if v == 1 else 0 for v in row) for row in block)
                    template_col = block_col
                    break
        
        if template_shape is None:
            continue
        
        # Fill other all-zero blocks
        for block_col in block_positions:
            if block_col == template_col or block_col + 3 > cols:
                continue
            
            is_all_zeros = all(grid[r][c] == 0 for r in range(row_start, row_end) for c in range(block_col, block_col + 3))
            
            if is_all_zeros:
                for r in range(row_start, row_end):
                    for c in range(block_col, block_col + 3):
                        shape_r = r - row_start
                        shape_c = c - block_col
                        if template_shape[shape_r][shape_c] == 1:
                            result[r][c] = 8
    
    return result


if __name__ == "__main__":
    # Load task
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/5b526a93.json') as f:
        task = json.load(f)
    
    # Test all training examples
    all_pass = True
    for i, example in enumerate(task['train']):
        inp = example['input']
        expected = example['output']
        result = solve(inp)
        
        # Compare
        match = result == expected
        status = "PASS" if match else "FAIL"
        print(f"Training example {i}: {status}")
        
        if not match:
            all_pass = False
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape: {len(result)}x{len(result[0])}")
            
            # Show first difference
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  First diff at ({r}, {c}): got {result[r][c]}, expected {expected[r][c]}")
                        break
                else:
                    continue
                break
    
    if all_pass:
        print("\nAll training examples passed!")
    else:
        print("\nSome examples failed!")
