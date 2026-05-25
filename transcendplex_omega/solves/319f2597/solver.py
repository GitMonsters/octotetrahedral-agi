def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Solves ARC-AGI task 319f2597.
    
    Rule: Find 2x2 blocks of zeros. For each such block at position (r,c):
    1. Set all values in rows r and r+1 to 0 (except preserve 2s)
    2. Set all values in columns c and c+1 to 0 (except preserve 2s)
    3. Values of 2 in the affected rows/columns remain as 2
    """
    # Create a copy of the input grid
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Find all 2x2 blocks of zeros
    zero_blocks = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid[r][c] == 0 and grid[r][c+1] == 0 and 
                grid[r+1][c] == 0 and grid[r+1][c+1] == 0):
                zero_blocks.append((r, c))
    
    # For each 2x2 zero block, extend zeros in horizontal and vertical lines
    for r, c in zero_blocks:
        # Extend horizontally through the entire row for both rows of the 2x2 block
        for col in range(cols):
            if grid[r][col] != 2:  # Preserve 2s
                result[r][col] = 0
            if grid[r+1][col] != 2:  # Preserve 2s  
                result[r+1][col] = 0
        
        # Extend vertically through the entire column for both columns of the 2x2 block  
        for row in range(rows):
            if grid[row][c] != 2:  # Preserve 2s
                result[row][c] = 0
            if grid[row][c+1] != 2:  # Preserve 2s
                result[row][c+1] = 0
    
    return result

if __name__ == "__main__":
    import json
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/319f2597.json") as f:
        task = json.load(f)
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        status = "PASS" if result == ex["output"] else "FAIL"
        print(f"Train {i}: {status}")