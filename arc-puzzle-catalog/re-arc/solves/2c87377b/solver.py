def transform(input_grid):
    import copy
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Find cross: one row all same color, one col all same color
    h_row = v_col = cross_color = None
    for r in range(rows):
        vals = set(input_grid[r])
        if len(vals) == 1:
            h_row = r
            cross_color = input_grid[r][0]
            break
    
    for c in range(cols):
        if all(input_grid[r][c] == cross_color for r in range(rows)):
            v_col = c
            break
    
    # Background color (any non-cross cell not on cross lines)
    bg = None
    for r in range(rows):
        for c in range(cols):
            if r != h_row and c != v_col:
                bg = input_grid[r][c]
                break
        if bg is not None:
            break
    
    # Find markers (cells not bg and not cross_color, not on cross lines)
    markers = []
    for r in range(rows):
        for c in range(cols):
            if r == h_row or c == v_col:
                continue
            if input_grid[r][c] != bg:
                markers.append((r, c))
    
    N = len(markers)
    
    if N > 0:
        # Determine quadrant of markers relative to cross
        below = all(r > h_row for r, c in markers)
        above = all(r < h_row for r, c in markers)
        left = all(c < v_col for r, c in markers)
        right = all(c > v_col for r, c in markers)
        
        # Shift direction is AWAY from the marker quadrant
        dh = 0
        dv = 0
        if below:
            dh = -N  # move up
        elif above:
            dh = N   # move down
        if left:
            dv = N   # move right
        elif right:
            dv = -N  # move left
        
        new_h = h_row + dh
        new_v = v_col + dv
    else:
        # No markers: find diagonal shift that makes a quadrant square
        # Try all 4 directions, find minimum D
        best = None
        for sh, sv in [(1, -1), (1, 1), (-1, 1), (-1, -1)]:
            # For TL to be square: (h_row + sh*D) = (v_col + sv*D)
            # D*(sh - sv) = v_col - h_row
            denom = sh - sv
            if denom != 0:
                D_tl = (v_col - h_row) / denom
                if D_tl > 0 and D_tl == int(D_tl):
                    D_tl = int(D_tl)
                    nh = h_row + sh * D_tl
                    nv = v_col + sv * D_tl
                    if 0 <= nh < rows and 0 <= nv < cols:
                        if best is None or D_tl < best[0]:
                            best = (D_tl, nh, nv)
            
            # For BR to be square: (rows-1-h_row - sh*D) = (cols-1-v_col - sv*D)
            # -sh*D + sv*D = (cols-1-v_col) - (rows-1-h_row)
            # D*(sv - sh) = (cols-1-v_col) - (rows-1-h_row)
            denom2 = sv - sh
            if denom2 != 0:
                D_br = ((cols - 1 - v_col) - (rows - 1 - h_row)) / denom2
                if D_br > 0 and D_br == int(D_br):
                    D_br = int(D_br)
                    nh = h_row + sh * D_br
                    nv = v_col + sv * D_br
                    if 0 <= nh < rows and 0 <= nv < cols:
                        if best is None or D_br < best[0]:
                            best = (D_br, nh, nv)
            
            # For TR to be square: (h_row + sh*D) = (cols-1-v_col - sv*D)
            denom3 = sh + sv
            if denom3 != 0:
                D_tr = (cols - 1 - v_col - h_row) / denom3
                if D_tr > 0 and D_tr == int(D_tr):
                    D_tr = int(D_tr)
                    nh = h_row + sh * D_tr
                    nv = v_col + sv * D_tr
                    if 0 <= nh < rows and 0 <= nv < cols:
                        if best is None or D_tr < best[0]:
                            best = (D_tr, nh, nv)
            
            # For BL to be square: (rows-1-h_row - sh*D) = (v_col + sv*D)
            denom4 = -(sh + sv)
            if denom4 != 0:
                D_bl = (v_col - (rows - 1 - h_row)) / denom4
                if D_bl > 0 and D_bl == int(D_bl):
                    D_bl = int(D_bl)
                    nh = h_row + sh * D_bl
                    nv = v_col + sv * D_bl
                    if 0 <= nh < rows and 0 <= nv < cols:
                        if best is None or D_bl < best[0]:
                            best = (D_bl, nh, nv)
        
        if best:
            _, new_h, new_v = best
        else:
            new_h, new_v = h_row, v_col
    
    # Build output: bg everywhere, then draw cross at new position
    output = [[bg for _ in range(cols)] for _ in range(rows)]
    
    # Draw horizontal line
    for c in range(cols):
        output[new_h][c] = cross_color
    
    # Draw vertical line
    for r in range(rows):
        output[r][new_v] = cross_color
    
    return output
