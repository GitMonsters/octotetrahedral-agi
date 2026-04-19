def transform(grid):
    """
    This puzzle has a repeating tile pattern. Some cells are corrupted.
    We need to find the tile period and reconstruct the correct pattern.
    """
    from collections import Counter
    
    grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    
    def find_periods(g):
        """Find row and column periods by checking pattern repetition."""
        # Find row period
        row_period = None
        for period in range(2, rows // 2 + 1):
            # Count how many row pairs match at this period
            matches = 0
            total = 0
            for i in range(rows - period):
                total += 1
                if g[i] == g[i + period]:
                    matches += 1
            # If most rows match, this is likely the period
            if total > 0 and matches >= total * 0.5:
                row_period = period
                break
        
        # Find column period
        col_period = None
        for period in range(2, cols // 2 + 1):
            matches = 0
            total = 0
            for j in range(cols - period):
                total += 1
                col_matches = True
                for i in range(rows):
                    if g[i][j] != g[i][j + period]:
                        col_matches = False
                        break
                if col_matches:
                    matches += 1
            if total > 0 and matches >= total * 0.5:
                col_period = period
                break
        
        return row_period, col_period
    
    # Try to find periods from partial matches in the input
    row_period, col_period = find_periods(grid)
    
    # If periods not found, try alternative detection
    if row_period is None or col_period is None:
        # Look for separator rows (all same value)
        sep_rows = []
        for i, row in enumerate(grid):
            if len(set(row)) == 1:
                sep_rows.append(i)
        
        # Look for separator columns
        sep_cols = []
        for j in range(cols):
            col_vals = [grid[i][j] for i in range(rows)]
            if len(set(col_vals)) == 1:
                sep_cols.append(j)
        
        # Determine periods from separators
        if len(sep_rows) >= 2:
            row_period = sep_rows[1] - sep_rows[0]
        elif len(sep_rows) == 1:
            row_period = max(sep_rows[0], rows - sep_rows[0] - 1, 2)
        
        if len(sep_cols) >= 2:
            col_period = sep_cols[1] - sep_cols[0]
        elif len(sep_cols) == 1:
            col_period = max(sep_cols[0], cols - sep_cols[0] - 1, 2)
    
    # Fallback
    if row_period is None:
        row_period = rows
    if col_period is None:
        col_period = cols
    
    # Build a reference tile by finding consensus for each position
    tile = [[None for _ in range(col_period)] for _ in range(row_period)]
    
    for ti in range(row_period):
        for tj in range(col_period):
            # Collect all values at this tile position across all repetitions
            values = []
            for block_row in range(0, rows, row_period):
                for block_col in range(0, cols, col_period):
                    gi = block_row + ti
                    gj = block_col + tj
                    if gi < rows and gj < cols:
                        values.append(grid[gi][gj])
            
            # The most common value is likely correct
            # But we need to handle the case where corruption is dominant
            counts = Counter(values)
            
            # If there's a clear winner (appears more than once or is unique), use it
            if len(counts) == 1:
                tile[ti][tj] = counts.most_common(1)[0][0]
            else:
                # Multiple candidates - pick the one that appears most
                tile[ti][tj] = counts.most_common(1)[0][0]
    
    # Now tile the entire grid with the reference tile
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            ti = i % row_period
            tj = j % col_period
            result[i][j] = tile[ti][tj]
    
    return result
