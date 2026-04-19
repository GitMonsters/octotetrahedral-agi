def transform(input_grid):
    from collections import Counter
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    all_vals = [v for row in input_grid for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            v = input_grid[r][c]
            if v != bg:
                color_positions.setdefault(v, []).append((r, c))
    
    seed_color = None
    seed_center = None
    pattern_cells = []
    
    non_bg_colors = list(color_positions.keys())
    
    if len(non_bg_colors) >= 2:
        # Find rectangular blocks, pick the LARGEST as seed
        rect_candidates = []
        for color in non_bg_colors:
            pos = color_positions[color]
            r0 = min(r for r, c in pos)
            r1 = max(r for r, c in pos)
            c0 = min(c for r, c in pos)
            c1 = max(c for r, c in pos)
            if len(pos) == (r1 - r0 + 1) * (c1 - c0 + 1):
                rect_candidates.append((len(pos), color, r0, r1, c0, c1))
        
        if rect_candidates:
            rect_candidates.sort(reverse=True)  # largest first
            _, seed_color, r0, r1, c0, c1 = rect_candidates[0]
            seed_center = ((r0 + r1) / 2, (c0 + c1) / 2)
        
        for color in non_bg_colors:
            if color != seed_color:
                for r, c in color_positions[color]:
                    pattern_cells.append((r, c, color))
    
    if seed_center is None:
        # Single non-bg color: find center at a corner of bbox
        all_pos = []
        for color in non_bg_colors:
            for r, c in color_positions[color]:
                all_pos.append((r, c))
                pattern_cells.append((r, c, color))
        
        r0 = min(r for r, c in all_pos)
        r1 = max(r for r, c in all_pos)
        c0 = min(c for r, c in all_pos)
        c1 = max(c for r, c in all_pos)
        
        corners = [
            (r0 - 0.5, c0 - 0.5),
            (r0 - 0.5, c1 + 0.5),
            (r1 + 0.5, c0 - 0.5),
            (r1 + 0.5, c1 + 0.5),
        ]
        
        for cr, cc in corners:
            fits = True
            all_on_bg = True
            for r, c, _ in pattern_cells:
                for nr, nc in [
                    (r, int(2 * cc - c)),
                    (int(2 * cr - r), c),
                    (int(2 * cr - r), int(2 * cc - c)),
                ]:
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        fits = False
                        break
                    if input_grid[nr][nc] != bg:
                        all_on_bg = False
                if not fits:
                    break
            if fits and all_on_bg:
                seed_center = (cr, cc)
                break
    
    if seed_center is None:
        return input_grid
    
    cr, cc = seed_center
    output = [row[:] for row in input_grid]
    
    for r, c, color in pattern_cells:
        for nr, nc in [
            (r, int(2 * cc - c)),
            (int(2 * cr - r), c),
            (int(2 * cr - r), int(2 * cc - c)),
        ]:
            if 0 <= nr < rows and 0 <= nc < cols:
                output[nr][nc] = color
    
    return output
