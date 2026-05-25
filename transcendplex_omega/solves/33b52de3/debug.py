def solve(grid: list[list[int]]) -> list[list[int]]:
    """Debug version with prints"""
    import copy
    result = copy.deepcopy(grid)
    
    # Find the key pattern
    key_positions = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != 0 and grid[i][j] != 5:
                key_positions.append((i, j))
    
    if not key_positions:
        return result
    
    # Get key bounds and build key matrix
    key_min_row = min(pos[0] for pos in key_positions)
    key_max_row = max(pos[0] for pos in key_positions)
    key_min_col = min(pos[1] for pos in key_positions)
    key_max_col = max(pos[1] for pos in key_positions)
    
    print(f"Key bounds: rows {key_min_row}-{key_max_row}, cols {key_min_col}-{key_max_col}")
    
    # Build the key matrix
    key_matrix = {}
    for i in range(key_min_row, key_max_row + 1):
        for j in range(key_min_col, key_max_col + 1):
            key_row = i - key_min_row
            key_col = j - key_min_col
            key_matrix[(key_row, key_col)] = grid[i][j]
    
    print("Key matrix:")
    for r in range(3):
        row_vals = []
        for c in range(5):
            row_vals.append(key_matrix.get((r, c), 0))
        print(f"  Row {r}: {row_vals}")
    
    # Test a few specific transformations
    test_positions = [(1, 2), (1, 3), (1, 6), (1, 14)]
    for i, j in test_positions:
        if i < len(grid) and j < len(grid[0]) and grid[i][j] == 5:
            block_col_idx = (j - 2) // 4
            in_block_col = (j - 2) % 4
            block_row_idx = (i - 1) // 4
            in_block_row = (i - 1) % 4
            
            print(f"Position ({i},{j}): block_col_idx={block_col_idx}, in_block_row={in_block_row}")
            if (in_block_row, block_col_idx) in key_matrix:
                new_val = key_matrix[(in_block_row, block_col_idx)]
                print(f"  -> Should become {new_val}")
            else:
                print(f"  -> Key position ({in_block_row}, {block_col_idx}) not found")
    
    return result

# Test with first example
import json
with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/33b52de3.json") as f:
    task = json.load(f)

example = task["train"][0]
result = solve(example["input"])