def transform(grid):
    import numpy as np
    from collections import Counter
    
    grid = [list(row) for row in grid]
    inp = np.array(grid)
    R, C = inp.shape
    
    flat = [int(inp[r][c]) for r in range(R) for c in range(C)]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find divider lines (non-bg first, then bg)
    div_r, div_r_color = None, None
    div_c, div_c_color = None, None
    
    for r in range(R):
        vals = set(int(v) for v in inp[r])
        if len(vals) == 1:
            v = int(list(vals)[0])
            if v != bg:
                div_r, div_r_color = r, v
                break
    
    if div_r is None:
        for r in range(R):
            vals = set(int(v) for v in inp[r])
            if len(vals) == 1:
                top_h, bot_h = r, R - r - 1
                if 1 <= min(top_h, bot_h) <= 3 and max(top_h, bot_h) >= 5:
                    div_r, div_r_color = r, int(list(vals)[0])
                    break
    
    for c in range(C):
        vals = set(int(inp[r][c]) for r in range(R))
        if len(vals) == 1:
            v = int(list(vals)[0])
            if v != bg:
                div_c, div_c_color = c, v
                break
    
    if div_c is None:
        for c in range(C):
            vals = set(int(inp[r][c]) for r in range(R))
            if len(vals) == 1:
                left_w, right_w = c, C - c - 1
                if 1 <= min(left_w, right_w) <= 3 and max(left_w, right_w) >= 5:
                    div_c, div_c_color = c, int(list(vals)[0])
                    break
    
    if div_r is None or div_c is None:
        return grid
    
    # The divider color to replace
    div_color = div_r_color  # Use the row divider color
    
    # Four quadrants
    quads = {
        'TL': (0, div_r, 0, div_c),
        'TR': (0, div_r, div_c+1, C),
        'BL': (div_r+1, R, 0, div_c),
        'BR': (div_r+1, R, div_c+1, C),
    }
    
    # Find main (largest) and template
    quad_sizes = {n: (r1-r0)*(c1-c0) for n, (r0,r1,c0,c1) in quads.items()}
    main_name = max(quad_sizes, key=quad_sizes.get)
    
    template_name = None
    for name in sorted(quad_sizes, key=quad_sizes.get):
        if name == main_name:
            continue
        r0, r1, c0, c1 = quads[name]
        region = inp[r0:r1, c0:c1]
        unique = set(int(v) for v in region.flatten())
        if len(unique) >= 2:
            template_name = name
            break
    
    if template_name is None:
        for name in sorted(quad_sizes, key=quad_sizes.get):
            if name != main_name:
                template_name = name
                break
    
    mr0, mr1, mc0, mc1 = quads[main_name]
    tr0, tr1, tc0, tc1 = quads[template_name]
    
    main_area = inp[mr0:mr1, mc0:mc1].copy()
    template = inp[tr0:tr1, tc0:tc1].copy()
    
    mH, mW = main_area.shape
    tH, tW = template.shape
    
    sub_h = mH // tH
    sub_w = mW // tW
    
    # Check if main area has any divider-colored cells
    has_div_cells = np.any(main_area == div_color)
    
    # Determine which color to replace
    if has_div_cells:
        replace_color = div_color
    else:
        # Replace the bg/most-common color
        main_flat = [int(v) for v in main_area.flatten()]
        replace_color = Counter(main_flat).most_common(1)[0][0]
    
    out = main_area.copy()
    
    for ti in range(tH):
        for tj in range(tW):
            tcolor = int(template[ti][tj])
            r_start = ti * sub_h
            r_end = (ti + 1) * sub_h
            c_start = tj * sub_w
            c_end = (tj + 1) * sub_w
            
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    v = int(main_area[r][c])
                    if v == replace_color:
                        out[r][c] = tcolor
                    else:
                        out[r][c] = v
    
    return out.tolist()
