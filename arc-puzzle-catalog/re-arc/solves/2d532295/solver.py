from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    
    # Find marker color: exactly 4 pixels forming rectangle corners
    marker_color = None
    marker_positions = []
    for color, cnt in sorted(counts.items(), key=lambda x: x[1]):
        if cnt == 4:
            positions = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
            rs = sorted(set(r for r,c in positions))
            cs = sorted(set(c for r,c in positions))
            if len(rs) == 2 and len(cs) == 2:
                # Check all 4 corners exist
                if all((r,c) in [(p[0],p[1]) for p in positions] for r in rs for c in cs):
                    marker_color = color
                    marker_positions = positions
                    break
    
    bg = counts.most_common(1)[0][0]
    
    if marker_color is not None:
        # Extract interior
        rs = sorted(set(r for r,c in marker_positions))
        cs = sorted(set(c for r,c in marker_positions))
        r1, r2 = rs[0]+1, rs[1]-1
        c1, c2 = cs[0]+1, cs[1]-1
        
        # Find pattern color
        pattern_color = None
        for color in counts:
            if color != bg and color != marker_color:
                pattern_color = color
                break
        
        out = []
        for r in range(r1, r2+1):
            row = []
            for c in range(c1, c2+1):
                v = grid[r][c]
                if pattern_color is not None and v == pattern_color:
                    row.append(marker_color)
                else:
                    row.append(v)
            out.append(row)
        return out
    else:
        # No markers: find the pattern area, return all bg
        # Find the bounding box of the non-bg pattern
        non_bg = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
        if non_bg:
            min_r = min(r for r,c in non_bg)
            max_r = max(r for r,c in non_bg)
            min_c = min(c for r,c in non_bg)
            max_c = max(c for r,c in non_bg)
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            return [[bg]*w for _ in range(h)]
        else:
            return [row[:] for row in grid]

