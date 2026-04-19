from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    counts = Counter(c for r in grid for c in r)
    bg = counts.most_common(1)[0][0]
    
    top_bg = 0
    while top_bg < rows and all(grid[top_bg][c] == bg for c in range(cols)):
        top_bg += 1
    left_bg = 0
    while left_bg < cols and all(grid[r][left_bg] == bg for r in range(rows)):
        left_bg += 1
    bottom_bg = 0
    while bottom_bg < rows and all(grid[rows-1-bottom_bg][c] == bg for c in range(cols)):
        bottom_bg += 1
    right_bg = 0
    while right_bg < cols and all(grid[r][cols-1-right_bg] == bg for r in range(rows)):
        right_bg += 1
    
    r0, r1 = top_bg, rows - bottom_bg
    c0, c1 = left_bg, cols - right_bg
    
    pat = [[grid[r][c] for c in range(c0, c1)] for r in range(r0, r1)]
    
    def find_vp(p):
        h, w = len(p), len(p[0])
        for per in range(1, h):
            if all(p[r][c] == p[r%per][c] for r in range(h) for c in range(w)):
                return per
        return h
    def find_hp(p):
        h, w = len(p), len(p[0])
        for per in range(1, w):
            if all(p[r][c] == p[r][c%per] for r in range(h) for c in range(w)):
                return per
        return w
    
    vp = find_vp(pat)
    hp = find_hp(pat)
    tile = [[pat[r][c] for c in range(hp)] for r in range(vp)]
    
    phase_r = (vp - r0 % vp) % vp
    phase_c = (hp - c0 % hp) % hp
    
    borders = {}
    if top_bg > 1: borders['top'] = top_bg
    if bottom_bg > 1: borders['bottom'] = bottom_bg
    if left_bg > 1: borders['left'] = left_bg
    if right_bg > 1: borders['right'] = right_bg
    if not borders:
        if top_bg > 0: borders['top'] = top_bg
        if bottom_bg > 0: borders['bottom'] = bottom_bg
        if left_bg > 0: borders['left'] = left_bg
        if right_bg > 0: borders['right'] = right_bg
    
    if borders:
        direction = min(borders, key=borders.get)
        if direction == 'top': phase_r = (phase_r - 1) % vp
        elif direction == 'bottom': phase_r = (phase_r + 1) % vp
        elif direction == 'left': phase_c = (phase_c - 1) % hp
        elif direction == 'right': phase_c = (phase_c + 1) % hp
    
    return [[tile[(r + phase_r) % vp][(c + phase_c) % hp] for c in range(cols)] for r in range(rows)]
