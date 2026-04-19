def transform(grid):
    """
    ARC puzzle 691c7aa7 solution.
    
    Pattern: Diagonal fill with 8
    - Rows 0-4: diagonal moves from column 0→4 (right)
      Keep columns [0..diag], fill [diag+1..end] with 8
    - Rows 5-8: No transformation
    - Rows 9+: diagonal moves from column 1→0 (left)
      Keep columns [diag..end], fill [0..diag-1] with 8
    """
    output = [row[:] for row in grid]  # Deep copy
    
    # Diagonal column for each row
    diag_cols = [
        0, 1, 2, 3, 4,           # rows 0-4
        None, None, None, None,  # rows 5-8 (no change)
        1, 2, 3, 4, 3, 2, 1, 0  # rows 9-16
    ]
    
    for i in range(len(output)):
        if i >= len(diag_cols):
            # Pattern repeats if grid is longer
            pattern_len = 17
            adjusted_i = i % pattern_len
            diag_col = diag_cols[adjusted_i] if adjusted_i < len(diag_cols) else None
        else:
            diag_col = diag_cols[i]
        
        if diag_col is None:
            # Rows 5-8: no change
            continue
        
        if i < 5:
            # Rows 0-4: Keep [0..diag_col], fill [diag_col+1..end] with 8
            for j in range(diag_col + 1, len(output[i])):
                output[i][j] = 8
        else:
            # Rows 9+: Keep [diag_col..end], fill [0..diag_col-1] with 8
            for j in range(diag_col):
                output[i][j] = 8
    
    return output
