from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    oH, oW = 7, 6
    
    # Find background color
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Group non-bg cells by color
    color_groups = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                color_groups.setdefault(grid[r][c], []).append((r, c))
    
    # Try to find corner markers: 4 cells of same color forming corners of 7x6 rectangle
    rect = None
    corner_color = None
    for color, cells in color_groups.items():
        if len(cells) == 4:
            rs = sorted(set(x[0] for x in cells))
            cs = sorted(set(x[1] for x in cells))
            if len(rs) == 2 and len(cs) == 2:
                if rs[1] - rs[0] + 1 == oH and cs[1] - cs[0] + 1 == oW:
                    corners = {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}
                    if set(cells) == corners:
                        rect = (rs[0], cs[0], rs[1], cs[1])
                        corner_color = color
                        break
    
    def is_on_border(r, c, r1, c1, r2, c2):
        if r1 <= r <= r2 and c1 <= c <= c2:
            return r == r1 or r == r2 or c == c1 or c == c2
        return False
    
    def find_displacement(cells, r1, c1, r2, c2):
        """Find the unique displacement vector that maps all cells to the rectangle border."""
        if not cells:
            return None
        r0, c0 = cells[0]
        candidates = set()
        for nc in range(c1, c2 + 1):
            candidates.add((r1 - r0, nc - c0))
            candidates.add((r2 - r0, nc - c0))
        for nr in range(r1 + 1, r2):
            candidates.add((nr - r0, c1 - c0))
            candidates.add((nr - r0, c2 - c0))
        
        for dr, dc in candidates:
            valid = True
            for r, c in cells:
                if not is_on_border(r + dr, c + dc, r1, c1, r2, c2):
                    valid = False
                    break
            if valid:
                return (dr, dc)
        return None
    
    if rect:
        r1, c1, r2, c2 = rect
    else:
        found = False
        for r1 in range(H - oH + 1):
            for c1 in range(W - oW + 1):
                r2, c2 = r1 + oH - 1, c1 + oW - 1
                all_valid = True
                for color, cells in color_groups.items():
                    d = find_displacement(cells, r1, c1, r2, c2)
                    if d is None:
                        all_valid = False
                        break
                if all_valid:
                    rect = (r1, c1, r2, c2)
                    found = True
                    break
            if found:
                break
        if not found:
            return [[bg] * oW for _ in range(oH)]
        r1, c1, r2, c2 = rect
    
    # Build output grid
    output = [[bg] * oW for _ in range(oH)]
    
    for color, cells in color_groups.items():
        if color == corner_color:
            for r, c in cells:
                or_ = 0 if r == r1 else oH - 1
                oc = 0 if c == c1 else oW - 1
                output[or_][oc] = color
        else:
            d = find_displacement(cells, r1, c1, r2, c2)
            if d is None:
                continue
            dr, dc = d
            for r, c in cells:
                or_ = r + dr - r1
                oc = c + dc - c1
                output[or_][oc] = color
    
    return output
