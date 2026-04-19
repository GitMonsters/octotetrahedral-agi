def transform(grid):
    import numpy as np
    grid = [list(row) for row in grid]
    inp = np.array(grid)
    R, C = inp.shape
    
    # Find rectangle outline with uniform interior
    best = None
    for h in range(3, R+1):
        for w in range(3, C+1):
            for r0 in range(R - h + 1):
                for c0 in range(C - w + 1):
                    r1, c1 = r0 + h - 1, c0 + w - 1
                    border_color = int(inp[r0][c0])
                    ok = True
                    for r in range(r0, r1 + 1):
                        for c in range(c0, c1 + 1):
                            on_border = (r == r0 or r == r1 or c == c0 or c == c1)
                            if on_border and int(inp[r][c]) != border_color:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    interior = inp[r0+1:r1, c0+1:c1]
                    if interior.size == 0:
                        continue
                    unique = np.unique(interior)
                    if len(unique) == 1 and int(unique[0]) != border_color:
                        best = (h, w, r0, c0, border_color, int(unique[0]))
                        break
                if best:
                    break
            if best:
                break
        if best:
            break
    
    if best is None:
        return grid
    
    rh, rw, _, _, fill_color, target_val = best
    ih, iw = rh - 2, rw - 2
    
    # Find all matching interior positions
    matches = []
    for r in range(R - ih + 1):
        for c in range(C - iw + 1):
            if np.all(inp[r:r+ih, c:c+iw] == target_val):
                matches.append((r, c))
    
    # Greedy non-overlapping selection
    occupied = set()
    selected = []
    for r, c in matches:
        or0, oc0 = r - 1, c - 1
        or1, oc1 = r + ih, c + iw
        # Check if this rectangle overlaps with any occupied cell
        overlap = False
        for dr in range(max(0, or0), min(R, or1 + 1)):
            for dc in range(max(0, oc0), min(C, oc1 + 1)):
                if (dr, dc) in occupied:
                    overlap = True
                    break
            if overlap:
                break
        
        if not overlap:
            selected.append((r, c))
            for dr in range(max(0, or0), min(R, or1 + 1)):
                for dc in range(max(0, oc0), min(C, oc1 + 1)):
                    occupied.add((dr, dc))
    
    # Draw outlines
    out = inp.copy()
    for r, c in selected:
        or0, oc0 = r - 1, c - 1
        or1, oc1 = r + ih, c + iw
        for dr in range(or0, or1 + 1):
            for dc in range(oc0, oc1 + 1):
                if 0 <= dr < R and 0 <= dc < C:
                    on_border = (dr == or0 or dr == or1 or dc == oc0 or dc == oc1)
                    if on_border:
                        out[dr][dc] = fill_color
    
    return out.tolist()
