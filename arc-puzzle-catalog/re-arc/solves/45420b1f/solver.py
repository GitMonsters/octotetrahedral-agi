import copy
from collections import Counter

def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    output = copy.deepcopy(input_grid)
    
    # Find background (most common) and all colors
    cnt = Counter()
    for r in range(rows):
        for c in range(cols):
            cnt[input_grid[r][c]] += 1
    bg = cnt.most_common(1)[0][0]
    colors = sorted(cnt.keys())
    
    rect = None
    fg = None
    
    # Case 1: Three colors - find the one forming a solid rectangle
    if len(colors) >= 3:
        for v in colors:
            if v == bg:
                continue
            pos = [(r, c) for r in range(rows) for c in range(cols) if input_grid[r][c] == v]
            if len(pos) < 4:
                continue
            rmin = min(p[0] for p in pos)
            rmax = max(p[0] for p in pos)
            cmin = min(p[1] for p in pos)
            cmax = max(p[1] for p in pos)
            expected = (rmax - rmin + 1) * (cmax - cmin + 1)
            if expected == len(pos):
                is_rect = all(input_grid[r][c] == v for r in range(rmin, rmax+1) for c in range(cmin, cmax+1))
                if is_rect:
                    rect = (rmin, cmin, rmax, cmax)
                    fg = [c for c in colors if c != bg and c != v][0]
                    break
    
    # Case 2: Two colors - look for a solid rectangular block of non-bg color
    if rect is None:
        for v in colors:
            if v == bg:
                continue
            # Find largest solid rectangle of this color using histogram method
            heights = [0] * cols
            best_area = 0
            best_rect = None
            for r in range(rows):
                for c in range(cols):
                    heights[c] = heights[c] + 1 if input_grid[r][c] == v else 0
                stack = []
                for c in range(cols + 1):
                    h = heights[c] if c < cols else 0
                    while stack and heights[stack[-1]] > h:
                        height = heights[stack.pop()]
                        width = c if not stack else c - stack[-1] - 1
                        area = height * width
                        if area > best_area:
                            best_area = area
                            c_start = 0 if not stack else stack[-1] + 1
                            r_start = r - height + 1
                            best_rect = (r_start, c_start, r, c_start + width - 1)
                    stack.append(c)
            if best_area >= 12 and best_rect is not None:
                r1, c1, r2, c2 = best_rect
                # Verify it's a solid block
                is_solid = all(input_grid[r][c] == v for r in range(r1, r2+1) for c in range(c1, c2+1))
                if is_solid:
                    rect = best_rect
                    fg = v
                    break
    
    # Case 3: Invisible rectangle (same color as background)
    if rect is None:
        fg = [v for v in colors if v != bg][0]
        rect = _find_invisible_rect(input_grid, bg, fg, rows, cols)
    
    if rect is None:
        return output
    
    # Apply projection rule
    r1, c1, r2, c2 = rect
    
    # Horizontal projections on rect rows
    for r in range(r1, r2 + 1):
        # Dots to the left of rect
        for c in range(0, c1):
            if input_grid[r][c] == fg:
                for cc in range(c, c1):
                    output[r][cc] = fg
        # Dots to the right of rect
        for c in range(c2 + 1, cols):
            if input_grid[r][c] == fg:
                for cc in range(c2 + 1, c + 1):
                    output[r][cc] = fg
    
    # Vertical projections on rect columns
    for c in range(c1, c2 + 1):
        # Dots above rect
        for r in range(0, r1):
            if input_grid[r][c] == fg:
                for rr in range(r, r1):
                    output[rr][c] = fg
        # Dots below rect
        for r in range(r2 + 1, rows):
            if input_grid[r][c] == fg:
                for rr in range(r2 + 1, r + 1):
                    output[rr][c] = fg
    
    return output


def _find_invisible_rect(grid, bg, fg, rows, cols):
    """Find invisible rectangle (same color as background) by brute force scoring."""
    import math
    
    # Precompute fg positions
    fg_by_row = {r: [c for c in range(cols) if grid[r][c] == fg] for r in range(rows)}
    fg_by_col = {c: [r for r in range(rows) if grid[r][c] == fg] for c in range(cols)}
    
    # Prefix sum for fast empty-rect check
    prefix = [[0]*(cols+1) for _ in range(rows+1)]
    for r in range(rows):
        for c in range(cols):
            prefix[r+1][c+1] = prefix[r][c+1] + prefix[r+1][c] - prefix[r][c] + (1 if grid[r][c] == fg else 0)
    
    def count_fg_in(r1, c1, r2, c2):
        return prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]
    
    def get_side_dists(r1, c1, r2, c2):
        dists = []
        left = [c1 - c for r in range(r1, r2+1) for c in fg_by_row[r] if c < c1]
        dists.append(min(left) if left else 999)
        right = [c - c2 for r in range(r1, r2+1) for c in fg_by_row[r] if c > c2]
        dists.append(min(right) if right else 999)
        top = [r1 - r for c in range(c1, c2+1) for r in fg_by_col[c] if r < r1]
        dists.append(min(top) if top else 999)
        bot = [r - r2 for c in range(c1, c2+1) for r in fg_by_col[c] if r > r2]
        dists.append(min(bot) if bot else 999)
        return dists
    
    grid_cr = (rows - 1) / 2
    grid_cc = (cols - 1) / 2
    
    best = None
    best_score = None
    
    for r1 in range(rows - 4):
        for r2 in range(r1 + 4, min(r1 + 7, rows)):
            for c1 in range(cols - 4):
                for c2 in range(c1 + 4, min(c1 + 7, cols)):
                    if count_fg_in(r1, c1, r2, c2) > 0:
                        continue
                    
                    dists = get_side_dists(r1, c1, r2, c2)
                    if any(d >= 999 for d in dists):
                        continue
                    if min(dists) < 2:
                        continue
                    
                    h, w = r2 - r1 + 1, c2 - c1 + 1
                    area = h * w
                    
                    total_proj = 0
                    for r in range(r1, r2+1):
                        total_proj += sum(1 for c in fg_by_row[r] if c < c1 or c > c2)
                    for c in range(c1, c2+1):
                        total_proj += sum(1 for r in fg_by_col[c] if r < r1 or r > r2)
                    
                    cr, cc = (r1+r2)/2, (c1+c2)/2
                    dist_center = math.sqrt((cr-grid_cr)**2 + (cc-grid_cc)**2)
                    
                    # Score: prefer square-ish rects with many projections
                    aspect = max(h, w) / min(h, w)
                    adj_proj = total_proj / aspect
                    score = (adj_proj, -dist_center, min(dists))
                    
                    if best_score is None or score > best_score:
                        best_score = score
                        best = (r1, c1, r2, c2)
    
    return best
