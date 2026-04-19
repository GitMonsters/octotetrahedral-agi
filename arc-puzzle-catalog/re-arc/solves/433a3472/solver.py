def transform(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])
    OH, OW = 2 * H, 2 * W
    
    # Find non-7 pattern bounding box
    min_r, max_r, min_c, max_c = H, 0, W, 0
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != 7:
                min_r = min(min_r, r); max_r = max(max_r, r)
                min_c = min(min_c, c); max_c = max(max_c, c)
    
    ph = max_r - min_r + 1
    pw = max_c - min_c + 1
    P = [input_grid[r][min_c:max_c+1] for r in range(min_r, max_r+1)]
    
    output = []
    for r in range(OH):
        row = []
        for c in range(OW):
            dc = OW - 1 - c
            kr = r // ph
            kc = dc // pw
            pr = r % ph
            pc_in_P = pw - 1 - (dc % pw)
            
            if kr == kc:
                val = P[pr][pc_in_P]
            elif kr < kc:
                val = P[0][pc_in_P]
            else:
                val = P[pr][pw - 1]
            row.append(val)
        output.append(row)
    return output
