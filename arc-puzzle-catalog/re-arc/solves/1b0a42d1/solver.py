def transform(input_grid):
    from collections import Counter
    R = len(input_grid)
    C = len(input_grid[0])
    
    all_vals = Counter(input_grid[r][c] for r in range(R) for c in range(C))
    bg = all_vals.most_common(1)[0][0]
    
    non_bg_rows = [r for r in range(R) if any(input_grid[r][c] != bg for c in range(C))]
    non_bg_cols = [c for c in range(C) if any(input_grid[r][c] != bg for r in range(R))]
    
    output = [row[:] for row in input_grid]
    
    def find_period(indices, max_dim, get_data):
        """Find the tiling period for a set of non-bg indices."""
        if not indices:
            return None, None
        start = indices[0]
        span = indices[-1] - indices[0] + 1
        
        # Try periods from 1 to span
        for period in range(1, span + 1):
            match = True
            for idx in indices:
                src = start + ((idx - start) % period)
                if get_data(idx) != get_data(src):
                    match = False
                    break
            if match:
                # Verify the period covers at least one full pattern
                return start, period
        return start, span
    
    # Try row tiling first (more non-bg rows than cols typically)
    row_span = (non_bg_rows[-1] - non_bg_rows[0] + 1) if non_bg_rows else 0
    col_span = (non_bg_cols[-1] - non_bg_cols[0] + 1) if non_bg_cols else 0
    
    # Determine if row or column tiling based on which has more blank space to fill
    blank_rows = R - len(non_bg_rows)
    blank_cols = C - len(non_bg_cols)
    
    if blank_rows >= blank_cols and non_bg_rows:
        # Try row tiling
        start, period = find_period(non_bg_rows, R, lambda r: tuple(input_grid[r]))
        if start is not None:
            for r in range(R):
                src = start + ((r - start) % period)
                output[r] = list(input_grid[src])
            return output
    
    if non_bg_cols:
        # Try column tiling
        start, period = find_period(non_bg_cols, C, 
                                     lambda c: tuple(input_grid[r][c] for r in range(R)))
        if start is not None:
            for r in range(R):
                for c in range(C):
                    src_c = start + ((c - start) % period)
                    output[r][c] = input_grid[r][src_c]
            return output
    
    if non_bg_rows:
        start, period = find_period(non_bg_rows, R, lambda r: tuple(input_grid[r]))
        if start is not None:
            for r in range(R):
                src = start + ((r - start) % period)
                output[r] = list(input_grid[src])
            return output
    
    return output
