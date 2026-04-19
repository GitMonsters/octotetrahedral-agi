def transform(input_grid):
    import copy
    grid = [row[:] for row in input_grid]
    nrows = len(grid)
    ncols = len(grid[0])
    
    from collections import Counter
    colors = Counter()
    for r in range(nrows):
        for c in range(ncols):
            colors[grid[r][c]] += 1
    bg = colors.most_common(1)[0][0]
    
    # Find horizontal stripe (contiguous full rows of same non-bg color)
    h_stripe = None
    r = 0
    while r < nrows:
        row_set = set(grid[r])
        if len(row_set) == 1 and grid[r][0] != bg:
            color = grid[r][0]
            end_r = r
            while end_r + 1 < nrows and set(grid[end_r+1]) == {color}:
                end_r += 1
            if end_r - r + 1 >= 2:
                h_stripe = (r, end_r, color)
                break
            r = end_r + 1
        else:
            r += 1
    
    # Find vertical stripe (contiguous full cols of same non-bg color)
    v_stripe = None
    c = 0
    while c < ncols:
        col_set = set(grid[r][c] for r in range(nrows))
        if len(col_set) == 1 and grid[0][c] != bg:
            color = grid[0][c]
            end_c = c
            while end_c + 1 < ncols and set(grid[r][end_c+1] for r in range(nrows)) == {color}:
                end_c += 1
            if end_c - c + 1 >= 2:
                v_stripe = (c, end_c, color)
                break
            c = end_c + 1
        else:
            c += 1
    
    # Find dots: cells that are not bg and not part of the stripe
    dots = []
    for r in range(nrows):
        for c in range(ncols):
            v = grid[r][c]
            if v == bg:
                continue
            in_stripe = False
            if h_stripe and h_stripe[0] <= r <= h_stripe[1]:
                in_stripe = True
            if v_stripe and v_stripe[0] <= c <= v_stripe[1]:
                in_stripe = True
            if not in_stripe:
                dots.append((r, c))
    
    if dots and (h_stripe or v_stripe):
        # CASE A: dots + stripe -> reflect dots across stripe
        # Sort dots to find the line direction
        dots.sort()
        # Direction from first to last dot
        dr = 1 if dots[-1][0] > dots[0][0] else (-1 if dots[-1][0] < dots[0][0] else 0)
        dc = 1 if dots[-1][1] > dots[0][1] else (-1 if dots[-1][1] < dots[0][1] else 0)
        
        # Find apex: closest dot to stripe
        if h_stripe:
            stripe_min_r, stripe_max_r = h_stripe[0], h_stripe[1]
            # Distance from each dot to stripe boundary
            def dist_to_stripe(d):
                if d[0] < stripe_min_r:
                    return stripe_min_r - d[0]
                elif d[0] > stripe_max_r:
                    return d[0] - stripe_max_r
                return 0
            apex = min(dots, key=dist_to_stripe)
            
            # Gray direction: from apex along the dots, away from stripe
            if apex[0] < stripe_min_r:
                # Dots above stripe, away = upward (row decreasing)
                gray_dr = -abs(dr)
            else:
                # Dots below stripe, away = downward (row increasing)
                gray_dr = abs(dr)
            # Determine col direction of gray arm from apex
            # Find the next dot after apex going away from stripe
            gray_dc = dc if dots[-1] != apex else -dc
            # More robust: look at the actual dot positions
            other_dots = [d for d in dots if d != apex]
            if other_dots:
                next_dot = min(other_dots, key=lambda d: abs(d[0]-apex[0]) + abs(d[1]-apex[1]))
                gray_dr = 1 if next_dot[0] > apex[0] else -1
                gray_dc = 1 if next_dot[1] > apex[1] else -1
            
            # Extension direction: negate the parallel component (col for horizontal stripe)
            ext_dr = gray_dr
            ext_dc = -gray_dc
            
        elif v_stripe:
            stripe_min_c, stripe_max_c = v_stripe[0], v_stripe[1]
            def dist_to_stripe(d):
                if d[1] < stripe_min_c:
                    return stripe_min_c - d[1]
                elif d[1] > stripe_max_c:
                    return d[1] - stripe_max_c
                return 0
            apex = min(dots, key=dist_to_stripe)
            
            other_dots = [d for d in dots if d != apex]
            if other_dots:
                next_dot = min(other_dots, key=lambda d: abs(d[0]-apex[0]) + abs(d[1]-apex[1]))
                gray_dr = 1 if next_dot[0] > apex[0] else -1
                gray_dc = 1 if next_dot[1] > apex[1] else -1
            
            # Extension direction: negate the parallel component (row for vertical stripe)
            ext_dr = -gray_dr
            ext_dc = gray_dc
        
        # Trace from apex in extension direction until hitting a wall
        r, c = apex
        while True:
            nr, nc = r + ext_dr, c + ext_dc
            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                break
            # Check if entering the stripe
            if h_stripe and h_stripe[0] <= nr <= h_stripe[1]:
                break
            if v_stripe and v_stripe[0] <= nc <= v_stripe[1]:
                break
            grid[nr][nc] = 9
            r, c = nr, nc
    
    elif not dots and (h_stripe or v_stripe):
        # CASE B: no dots + stripe -> create V in larger region
        if h_stripe:
            stripe_r0, stripe_r1 = h_stripe[0], h_stripe[1]
            top_size = stripe_r0  # rows 0 to stripe_r0-1
            bot_size = nrows - stripe_r1 - 1  # rows stripe_r1+1 to nrows-1
            
            if top_size >= bot_size:
                # Larger region is top
                region_r0, region_r1 = 0, stripe_r0 - 1
                smaller = bot_size
                # Start at top wall (row 0), direction toward stripe (1, 1)
                total = 2 * smaller - 1
                start_r = 0
                start_c = total
                ball_dr, ball_dc = 1, 1
            else:
                # Larger region is bottom
                region_r0, region_r1 = stripe_r1 + 1, nrows - 1
                smaller = top_size
                total = 2 * smaller - 1
                start_r = region_r1
                start_c = total
                ball_dr, ball_dc = -1, 1
            
            # Trace ball within the region for 'total' cells
            r, c = start_r, start_c
            dr, dc = ball_dr, ball_dc
            for step in range(total):
                grid[r][c] = 9
                if step < total - 1:
                    nr, nc = r + dr, c + dc
                    if nr < region_r0 or nr > region_r1:
                        dr = -dr
                        nr = r + dr
                    if nc < 0 or nc >= ncols:
                        dc = -dc
                        nc = c + dc
                    r, c = nr, nc
        
        elif v_stripe:
            stripe_c0, stripe_c1 = v_stripe[0], v_stripe[1]
            left_size = stripe_c0
            right_size = ncols - stripe_c1 - 1
            
            if left_size >= right_size:
                region_c0, region_c1 = 0, stripe_c0 - 1
                smaller = right_size
                total = 2 * smaller - 1
                start_c = 0
                start_r = total
                ball_dr, ball_dc = 1, 1
            else:
                region_c0, region_c1 = stripe_c1 + 1, ncols - 1
                smaller = left_size
                total = 2 * smaller - 1
                start_c = region_c1
                start_r = total
                ball_dr, ball_dc = 1, -1
            
            r, c = start_r, start_c
            dr, dc = ball_dr, ball_dc
            for step in range(total):
                grid[r][c] = 9
                if step < total - 1:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= nrows:
                        dr = -dr
                        nr = r + dr
                    if nc < region_c0 or nc > region_c1:
                        dc = -dc
                        nc = c + dc
                    r, c = nr, nc
    
    elif dots and not (h_stripe or v_stripe):
        # CASE C: dots + no stripe -> extend line to grid edge
        dots.sort()
        dr = 1 if dots[-1][0] > dots[0][0] else (-1 if dots[-1][0] < dots[0][0] else 0)
        dc = 1 if dots[-1][1] > dots[0][1] else (-1 if dots[-1][1] < dots[0][1] else 0)
        
        # Extend from last dot in the same direction
        r, c = dots[-1]
        while True:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                break
            grid[nr][nc] = 9
            r, c = nr, nc
        
        # Extend from first dot in the opposite direction
        r, c = dots[0]
        while True:
            nr, nc = r - dr, c - dc
            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                break
            grid[nr][nc] = 9
            r, c = nr, nc
    
    return grid
