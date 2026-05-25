def solve(grid):
    """
    ARC puzzle c8b7cc0f solver.
    
    Pattern:
    1. Find a rectangle bounded by 1s in the input
    2. Identify the key color (non-zero, non-1 color) that appears inside the rectangle
    3. Count how many times the key color appears inside the rectangle
    4. Create a 3x3 output grid
    5. Fill the output from top-left, row-by-row, with the key color for N cells
    6. Fill the remaining cells with 0
    """
    
    # Find the rectangle bounded by 1s
    ones_positions = [
        (i, j) 
        for i, row in enumerate(grid) 
        for j, cell in enumerate(row) 
        if cell == 1
    ]
    
    if not ones_positions:
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    min_r = min(p[0] for p in ones_positions)
    max_r = max(p[0] for p in ones_positions)
    min_c = min(p[1] for p in ones_positions)
    max_c = max(p[1] for p in ones_positions)
    
    # Extract content inside rectangle
    inner = []
    for i in range(min_r + 1, max_r):
        inner.append(grid[i][min_c + 1:max_c])
    
    # Find all non-zero, non-1 colors
    all_colors = set()
    for row in inner:
        for cell in row:
            if cell != 0 and cell != 1:
                all_colors.add(cell)
    
    if not all_colors:
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    # Assume there's only one key color (based on examples)
    key_color = list(all_colors)[0]
    
    # Count occurrences of key_color inside the rectangle
    count = sum(row.count(key_color) for row in inner)
    
    # Create 3x3 output filled from top-left
    output = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    # Fill first 'count' cells with key_color
    filled = 0
    for i in range(3):
        for j in range(3):
            if filled < count:
                output[i][j] = key_color
                filled += 1
            else:
                break
        if filled >= count:
            break
    
    return output
