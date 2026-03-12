def solve(grid: list[list[int]]) -> list[list[int]]:
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
    
    key_matrix = {}
    for i in range(key_min_row, key_max_row + 1):
        for j in range(key_min_col, key_max_col + 1):
            key_row = i - key_min_row
            key_col = j - key_min_col
            key_matrix[(key_row, key_col)] = grid[i][j]
    
    # Process each 5
    changes = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 5:
                if j >= 2:
                    block_col_idx = (j - 2) // 4
                    in_block_col = (j - 2) % 4
                    
                    if in_block_col < 3 and block_col_idx < 5:  # Ensure valid key column
                        if i >= 1:
                            block_row_idx = (i - 1) // 4
                            in_block_row = (i - 1) % 4
                            
                            if in_block_row < 3:
                                key_lookup = (in_block_row, block_col_idx)
                                if key_lookup in key_matrix:
                                    result[i][j] = key_matrix[key_lookup]
                                    changes += 1
    
    print(f"Made {changes} changes")
    return result

# Test
import json
with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/33b52de3.json") as f:
    task = json.load(f)

example = task["train"][0]
result = solve(example["input"])

# Compare a few specific positions
test_positions = [(1,2), (1,6), (1,14), (2,2), (3,3)]
print("Checking specific positions:")
for i, j in test_positions:
    if i < len(result) and j < len(result[0]):
        exp = example["output"][i][j]
        got = result[i][j]
        inp = example["input"][i][j]
        print(f"({i},{j}): input={inp}, expected={exp}, got={got}, {'✓' if exp==got else '✗'}")