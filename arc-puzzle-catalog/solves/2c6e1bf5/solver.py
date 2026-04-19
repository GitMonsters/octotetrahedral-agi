def transform(grid):
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background (most common color)
    cc = Counter()
    for r in range(rows):
        for c in range(cols):
            cc[grid[r][c]] += 1
    bg = cc.most_common(1)[0][0]
    
    # Find dots: isolated non-bg pixels
    dots = []
    dot_set = set()
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color == bg:
                continue
            same_count = 0
            bg_count = 0
            same_neighbor = None
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nv = grid[nr][nc]
                    if nv == color:
                        same_count += 1
                        same_neighbor = (nr, nc)
                    elif nv == bg:
                        bg_count += 1
                else:
                    bg_count += 1
            
            if same_count == 0:
                dots.append((r, c, color))
                dot_set.add((r, c))
            elif same_count == 1 and bg_count >= 3:
                # Spur detection: neighbor must have >=2 other same-color neighbors
                snr, snc = same_neighbor
                other_same = 0
                for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nnr, nnc = snr+dr2, snc+dc2
                    if (nnr, nnc) != (r, c) and 0 <= nnr < rows and 0 <= nnc < cols:
                        if grid[nnr][nnc] == color:
                            other_same += 1
                if other_same >= 2:
                    dots.append((r, c, color))
                    dot_set.add((r, c))
    
    # Active diagonals from all dots
    diag_plus = set()
    diag_minus = set()
    for dr, dc, _ in dots:
        diag_plus.add(dr + dc)
        diag_minus.add(dr - dc)
    
    # Classify dots into bg-dots and shape-dots
    bg_dots_colors = Counter()
    shape_dots_colors = Counter()
    for r, c, color in dots:
        bg_n = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                   if 0 <= r+dr < rows and 0 <= c+dc < cols and grid[r+dr][c+dc] == bg)
        non_bg_non_dot_n = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                               if 0 <= r+dr < rows and 0 <= c+dc < cols 
                               and grid[r+dr][c+dc] != bg and (r+dr, c+dc) not in dot_set)
        if bg_n > non_bg_non_dot_n:
            bg_dots_colors[color] += 1
        else:
            shape_dots_colors[color] += 1
    
    bg_alt = bg_dots_colors.most_common(1)[0][0] if bg_dots_colors else bg
    shape_alt = shape_dots_colors.most_common(1)[0][0] if shape_dots_colors else bg
    
    # Find shape base color
    shape_base = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r, c) not in dot_set:
                shape_base = grid[r][c]
                break
        if shape_base is not None:
            break
    if shape_base is None:
        shape_base = bg
    
    # Build output
    output = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if (r, c) in dot_set:
                output[r][c] = grid[r][c]
            else:
                on_diag = (r + c) in diag_plus or (r - c) in diag_minus
                if grid[r][c] == bg:
                    output[r][c] = bg_alt if on_diag else bg
                else:
                    output[r][c] = shape_alt if on_diag else shape_base
    
    return output
