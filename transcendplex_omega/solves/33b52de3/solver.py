def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC task 33b52de3 solver.
    
    Pattern: 
    1. Find the key pattern (small colored region, non-0 and non-5 values)
    2. Find the grid of 5s blocks 
    3. Each block position maps to corresponding key position
    4. Replace 5s with values from key based on block coordinates
    """
    import copy
    result = copy.deepcopy(grid)
    
    # Find the key pattern (non-0, non-5 colored region)
    key_positions = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != 0 and grid[i][j] != 5:
                key_positions.append((i, j))
    
    if not key_positions:
        return result
    
    # Build key matrix
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
    
    # Find starting position of the 5s block grid
    block_start_row = None
    block_start_col = None
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 5:
                block_start_row = i
                block_start_col = j
                break
        if block_start_row is not None:
            break
    
    if block_start_row is None:
        return result
    
    # Replace each 5 with the corresponding key value
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 5:
                # Calculate block coordinates (blocks spaced 4 units apart)
                block_row = (i - block_start_row) // 4
                block_col = (j - block_start_col) // 4
                
                # Map to key value if within bounds
                if (block_row, block_col) in key_matrix:
                    result[i][j] = key_matrix[(block_row, block_col)]
    
    return result


if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/33b52de3.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")