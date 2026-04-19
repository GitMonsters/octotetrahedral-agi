from collections import Counter

def transform(input_grid):
    grid = [row[:] for row in input_grid]
    H, W = len(grid), len(grid[0])
    
    bg = Counter(c for r in grid for c in r).most_common(1)[0][0]
    
    non_bg = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                non_bg.setdefault(grid[r][c], []).append((r, c))
    
    if not non_bg:
        return grid

    # Find rectangle (filled solid block)
    rect_color = None
    for color in sorted(non_bg.keys(), key=lambda k: -len(non_bg[k])):
        cells = non_bg[color]
        if len(cells) >= 4:
            rmin = min(r for r, c in cells)
            rmax = max(r for r, c in cells)
            cmin = min(c for r, c in cells)
            cmax = max(c for r, c in cells)
            if len(cells) == (rmax - rmin + 1) * (cmax - cmin + 1):
                rect_color = color
                break
    
    markers = []
    for color in non_bg:
        if color != rect_color:
            for (r, c) in non_bg[color]:
                markers.append((r, c, color))
    
    output = [row[:] for row in input_grid]
    
    if rect_color is not None and non_bg.get(rect_color):
        cells = non_bg[rect_color]
        rmin = min(r for r, c in cells)
        rmax = max(r for r, c in cells)
        cmin = min(c for r, c in cells)
        cmax = max(c for r, c in cells)
        rcr = (rmin + rmax) / 2
        rcc = (cmin + cmax) / 2
        
        # Project each marker to nearest rect edge
        projected = []
        for r, c, color in markers:
            if r < rmin:
                pr, pc, approach = rmin, c, 'top'
            elif r > rmax:
                pr, pc, approach = rmax, c, 'bottom'
            elif c < cmin:
                pr, pc, approach = r, cmin, 'left'
            elif c > cmax:
                pr, pc, approach = r, cmax, 'right'
            else:
                pr, pc, approach = r, c, 'inside'
            projected.append((pr, pc, color, approach))
            output[pr][pc] = color
        
        # Splash + Shadow for top/left approaching dots
        for pr, pc, color, approach in projected:
            if approach == 'top':
                # Splash: 1 cell horizontal toward center col
                if pc < rcc:
                    output[pr][pc + 1] = bg
                elif pc > rcc:
                    output[pr][pc - 1] = bg
                # Shadow: same col on bottom edge
                output[rmax][pc] = bg
            elif approach == 'left':
                # Splash: 1 cell vertical toward center row
                if pr < rcr:
                    output[pr + 1][pc] = bg
                elif pr > rcr:
                    output[pr - 1][pc] = bg
                # Shadow: same col on bottom edge
                output[rmax][pc] = bg
        
        # Triangle cut: if exactly 2 markers on 2 adjacent edges
        if len(projected) == 2:
            (pr1, pc1, col1, app1), (pr2, pc2, col2, app2) = projected
            
            # Check if on adjacent edges
            adjacent_pairs = {
                ('top', 'right'), ('right', 'top'),
                ('top', 'left'), ('left', 'top'),
                ('bottom', 'right'), ('right', 'bottom'),
                ('bottom', 'left'), ('left', 'bottom'),
            }
            
            if (app1, app2) in adjacent_pairs:
                # Find shared corner
                if set([app1, app2]) == {'top', 'right'}:
                    corner = (rmin, cmax)
                    opp_corner = (rmax, cmin)
                elif set([app1, app2]) == {'top', 'left'}:
                    corner = (rmin, cmin)
                    opp_corner = (rmax, cmax)
                elif set([app1, app2]) == {'bottom', 'right'}:
                    corner = (rmax, cmax)
                    opp_corner = (rmin, cmin)
                elif set([app1, app2]) == {'bottom', 'left'}:
                    corner = (rmax, cmin)
                    opp_corner = (rmin, cmax)
                else:
                    corner = None
                    opp_corner = None
                
                if corner:
                    # Walk perimeter from dot1 to dot2 via shared corner
                    def perimeter_walk(start, end, corner, rmin, rmax, cmin, cmax):
                        """Get cells on perimeter from start to end going through corner"""
                        # Build full perimeter path clockwise
                        perimeter = []
                        # Top edge left to right
                        for c in range(cmin, cmax + 1):
                            perimeter.append((rmin, c))
                        # Right edge top+1 to bottom
                        for r in range(rmin + 1, rmax + 1):
                            perimeter.append((r, cmax))
                        # Bottom edge right-1 to left
                        for c in range(cmax - 1, cmin - 1, -1):
                            perimeter.append((rmax, c))
                        # Left edge bottom-1 to top-1
                        for r in range(rmax - 1, rmin, -1):
                            perimeter.append((r, cmin))
                        
                        # Find indices
                        idx_start = perimeter.index(start)
                        idx_end = perimeter.index(end)
                        idx_corner = perimeter.index(corner)
                        
                        # Walk from start to end through corner (CW or CCW)
                        n = len(perimeter)
                        # Try CW
                        path_cw = []
                        i = (idx_start + 1) % n
                        found_corner = False
                        while i != idx_end:
                            path_cw.append(perimeter[i])
                            if perimeter[i] == corner:
                                found_corner = True
                            i = (i + 1) % n
                        
                        if found_corner:
                            return path_cw
                        
                        # Try CCW
                        path_ccw = []
                        i = (idx_start - 1) % n
                        found_corner = False
                        while i != idx_end:
                            path_ccw.append(perimeter[i])
                            if perimeter[i] == corner:
                                found_corner = True
                            i = (i - 1) % n
                        
                        if found_corner:
                            return path_ccw
                        
                        return path_cw  # fallback
                    
                    path = perimeter_walk((pr1, pc1), (pr2, pc2), corner, rmin, rmax, cmin, cmax)
                    
                    # Remove corner from bg cells, keep rest
                    for (r, c) in path:
                        if (r, c) != corner:
                            output[r][c] = bg
                    
                    # Opposite corner effects
                    output[opp_corner[0]][opp_corner[1]] = bg
                    
                    # Find top/left dot and right/bottom dot
                    if app1 in ('top', 'left'):
                        top_left_dot = (pr1, pc1)
                        other_dot = (pr2, pc2, app2)
                    else:
                        top_left_dot = (pr2, pc2)
                        other_dot = (pr1, pc1, app1)
                    
                    # Shadow of top/left dot already done
                    
                    # Reflect right/bottom dot across edge center to opposite edge
                    opr, opc, oapp = other_dot
                    if oapp == 'right':
                        # Reflect row across center of right edge
                        refl_r = int(2 * rcr - opr + 0.5)  # round
                        output[refl_r][cmin] = bg
                    elif oapp == 'bottom':
                        # Reflect col across center of bottom edge
                        refl_c = int(2 * rcc - opc + 0.5)
                        output[rmin][refl_c] = bg
                    elif oapp == 'left':
                        refl_r = int(2 * rcr - opr + 0.5)
                        output[refl_r][cmax] = bg
                    elif oapp == 'top':
                        refl_c = int(2 * rcc - opc + 0.5)
                        output[rmax][refl_c] = bg
        
        # Re-place dot colors (in case bg overwrote them)
        for pr, pc, color, approach in projected:
            output[pr][pc] = color
    
    else:
        # No rectangle: create copies
        by_color = {}
        for r, c, color in markers:
            by_color.setdefault(color, []).append((r, c))
        
        colors = list(by_color.keys())
        if len(colors) == 2:
            c1, c2 = colors
            dots1, dots2 = by_color[c1], by_color[c2]
            
            def get_shift(own_dots, other_dots):
                other_count = len(other_dots)
                own_cr = sum(r for r, c in own_dots) / len(own_dots)
                own_cc = sum(c for r, c in own_dots) / len(own_dots)
                other_cr = sum(r for r, c in other_dots) / len(other_dots)
                other_cc = sum(c for r, c in other_dots) / len(other_dots)
                
                if len(own_dots) > 1:
                    rs = max(r for r,c in own_dots) - min(r for r,c in own_dots)
                    cs = max(c for r,c in own_dots) - min(c for r,c in own_dots)
                    if rs >= cs:
                        return (0, 2 * other_count * (1 if other_cc > own_cc else -1))
                    else:
                        return (2 * other_count * (1 if other_cr > own_cr else -1), 0)
                else:
                    dr = abs(other_cr - own_cr)
                    dc = abs(other_cc - own_cc)
                    if dr >= dc:
                        return (2 * other_count * (1 if other_cr > own_cr else -1), 0)
                    else:
                        return (0, 2 * other_count * (1 if other_cc > own_cc else -1))
            
            s1 = get_shift(dots1, dots2)
            s2 = get_shift(dots2, dots1)
            
            for r, c in dots1:
                nr, nc = r + s1[0], c + s1[1]
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr][nc] = c1
            for r, c in dots2:
                nr, nc = r + s2[0], c + s2[1]
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr][nc] = c2
    
    return output
