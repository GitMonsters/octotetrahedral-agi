def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Find background color
    bg = Counter(v for row in input_grid for v in row).most_common(1)[0][0]
    
    # Collect markers on each edge
    left_m = {r: input_grid[r][0] for r in range(rows) if input_grid[r][0] != bg}
    right_m = {r: input_grid[r][cols-1] for r in range(rows) if input_grid[r][cols-1] != bg}
    top_m = {c: input_grid[0][c] for c in range(cols) if input_grid[0][c] != bg}
    bot_m = {c: input_grid[rows-1][c] for c in range(cols) if input_grid[rows-1][c] != bg}
    
    # Mode detection: count interior (non-corner) edge markers
    lr_count = sum(1 for r in range(1, rows-1) if input_grid[r][0] != bg or input_grid[r][cols-1] != bg)
    tb_count = sum(1 for c in range(1, cols-1) if input_grid[0][c] != bg or input_grid[rows-1][c] != bg)
    
    output = [row[:] for row in input_grid]
    
    if lr_count >= tb_count and lr_count > 0:
        # Left/Right mode
        midcol = cols // 2
        midrow = rows // 2
        use_ext = (rows % 2 == 1)
        
        # Fill left half (cols 0..midcol-1) for each left marker row
        for r, val in left_m.items():
            for c in range(midcol):
                output[r][c] = val
        
        # Fill right half (cols midcol+1..end) for each right marker row
        for r, val in right_m.items():
            for c in range(midcol + 1, cols):
                output[r][c] = val
        
        # Determine which rows get 7 at midcol
        seven = set(left_m.keys()) | set(right_m.keys())
        
        if use_ext:
            for markers in [left_m, right_m]:
                for r in markers:
                    # Top half: extend upward (away from midrow)
                    if 0 <= r < midrow and r != midrow - 1:
                        ext = r - 1
                        if ext >= 0:
                            seven.add(ext)
                    # Bottom half: extend downward (away from midrow)
                    elif midrow < r < rows and r != midrow + 1:
                        ext = r + 1
                        if ext < rows:
                            seven.add(ext)
        
        for r in seven:
            output[r][midcol] = 7
    
    else:
        # Top/Bottom mode
        midrow = rows // 2
        midcol = cols // 2
        use_ext = (cols % 2 == 1)
        
        # Fill top half (rows 0..midrow-1) for each top marker col
        for c, val in top_m.items():
            for r in range(midrow):
                output[r][c] = val
        
        # Fill bottom half (rows midrow+1..end) for each bottom marker col
        for c, val in bot_m.items():
            for r in range(midrow + 1, rows):
                output[r][c] = val
        
        # Determine which cols get 7 at midrow
        seven = set(top_m.keys()) | set(bot_m.keys())
        
        if use_ext:
            for markers in [top_m, bot_m]:
                for c in markers:
                    # Left half: extend leftward (away from midcol)
                    if 0 <= c < midcol and c != midcol - 1:
                        ext = c - 1
                        if ext >= 0:
                            seven.add(ext)
                    # Right half: extend rightward (away from midcol)
                    elif midcol < c < cols and c != midcol + 1:
                        ext = c + 1
                        if ext < cols:
                            seven.add(ext)
        
        for c in seven:
            output[midrow][c] = 7
    
    return output
