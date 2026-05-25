def solve(grid):
    """
    Solver for ARC puzzle fb791726.
    
    Transform algorithm:
    1. Find all vertical connections: columns where the same non-zero color appears in 2+ rows
    2. For each vertical connection (from row i1 to row i2):
       - Create a 3-row block: [row_i1_expanded | green_line | row_i2_expanded]
    3. Place first set of blocks at rows 0+ with original column positions
    4. Place second set of blocks at rows h+ with columns shifted right by w
    
    Output grid has double the dimensions (h*2 x w*2).
    """
    h, w = len(grid), len(grid[0])
    out_h = h * 2
    out_w = w * 2
    result = [[0] * out_w for _ in range(out_h)]
    
    # Find vertical connections by column
    connections = {}  # column -> list of (row, value) pairs
    for j in range(w):
        for i in range(h):
            if grid[i][j] != 0:
                if j not in connections:
                    connections[j] = []
                connections[j].append((i, grid[i][j]))
    
    # Sort connections by column
    sorted_cols = sorted(connections.keys())
    
    # Create blocks for each connection
    block_idx = 0
    for col in sorted_cols:
        pairs = connections[col]
        # Process consecutive pairs of non-empty rows in this column
        for pair_idx in range(len(pairs) - 1):
            i1, val1 = pairs[pair_idx]
            i2, val2 = pairs[pair_idx + 1]
            
            # First pass: original columns, starting at row 0
            out_row = block_idx * 3
            
            # Row i1 expanded to double width
            for j in range(w):
                result[out_row][j] = grid[i1][j]
            
            # Green line (color 3)
            out_row += 1
            for j in range(out_w):
                result[out_row][j] = 3
            
            # Row i2 expanded to double width
            out_row += 1
            for j in range(w):
                result[out_row][j] = grid[i2][j]
            
            # Second pass: shifted columns, starting at row h
            out_row = h + block_idx * 3
            
            # Row i1 with columns shifted right by w
            for j in range(w):
                result[out_row][j + w] = grid[i1][j]
            
            # Green line (color 3)
            out_row += 1
            for j in range(out_w):
                result[out_row][j] = 3
            
            # Row i2 with columns shifted right by w
            out_row += 1
            for j in range(w):
                result[out_row][j + w] = grid[i2][j]
            
            block_idx += 1
    
    return result
