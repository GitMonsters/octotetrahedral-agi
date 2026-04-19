from collections import Counter
import copy

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    cnt = Counter()
    for r in grid:
        for c in r:
            cnt[c] += 1
    
    colors = cnt.most_common()
    if len(colors) < 3:
        return [row[:] for row in grid]
    
    bg = colors[0][0]
    scatter = colors[1][0]
    border = colors[2][0]
    
    # Find bounding box of border color
    min_r, max_r, min_c, max_c = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == border:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    if min_r > max_r:
        return [row[:] for row in grid]
    
    template = []
    for r in range(min_r, max_r + 1):
        template.append([grid[r][c] for c in range(min_c, max_c + 1)])
    
    # Verify template has complete border of border color
    th, tw = len(template), len(template[0])
    
    def rotate90(g):
        h, w = len(g), len(g[0])
        return [[g[h - 1 - j][i] for j in range(h)] for i in range(w)]
    
    def flip_h(g):
        return [row[::-1] for row in g]
    
    # Generate all 8 orientations
    orientations = []
    seen = set()
    t = template
    for _ in range(4):
        for variant in [t, flip_h(t)]:
            key = tuple(tuple(row) for row in variant)
            if key not in seen:
                seen.add(key)
                orientations.append(variant)
        t = rotate90(t)
    
    # Try each orientation at each position
    result = [row[:] for row in grid]
    
    for tmpl in orientations:
        th, tw = len(tmpl), len(tmpl[0])
        for sr in range(rows - th + 1):
            for sc in range(cols - tw + 1):
                # Check if template can be placed here
                valid = True
                for tr in range(th):
                    for tc in range(tw):
                        tv = tmpl[tr][tc]
                        gv = grid[sr + tr][sc + tc]
                        if tv == border:
                            if gv != bg:
                                valid = False
                                break
                        elif tv == scatter:
                            if gv != scatter:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    # Place template
                    for tr in range(th):
                        for tc in range(tw):
                            if tmpl[tr][tc] == border:
                                result[sr + tr][sc + tc] = border
    
    return result
