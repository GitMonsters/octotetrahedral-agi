from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]
    
    output = [row[:] for row in grid]
    
    # Find non-bg pixels
    pixels = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                pixels.append((r, c, grid[r][c]))
    
    if len(pixels) == 2:
        (r1, c1, v1), (r2, c2, v2) = pixels
        
        if r1 == r2:
            # Same row, horizontal
            dist = abs(c2 - c1)
            bar_dist = (dist - 3) // 2
            # pixel 1 points right, pixel 2 points left (assuming c1 < c2)
            if c1 < c2:
                _draw_t(output, H, W, r1, c1, 0, 1, bar_dist, v1)
                _draw_t(output, H, W, r2, c2, 0, -1, bar_dist, v2)
            else:
                _draw_t(output, H, W, r1, c1, 0, -1, bar_dist, v1)
                _draw_t(output, H, W, r2, c2, 0, 1, bar_dist, v2)
        else:
            # Same col, vertical
            dist = abs(r2 - r1)
            bar_dist = (dist - 3) // 2
            if r1 < r2:
                _draw_t(output, H, W, r1, c1, 1, 0, bar_dist, v1)
                _draw_t(output, H, W, r2, c2, -1, 0, bar_dist, v2)
            else:
                _draw_t(output, H, W, r1, c1, -1, 0, bar_dist, v1)
                _draw_t(output, H, W, r2, c2, 1, 0, bar_dist, v2)
    
    elif len(pixels) == 1:
        r, c, v = pixels[0]
        # Find farthest edge
        dists = {'up': r, 'down': H - 1 - r, 'left': c, 'right': W - 1 - c}
        direction = max(dists, key=dists.get)
        dist = dists[direction]
        bar_dist = (dist - 3) // 2
        
        dr, dc = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}[direction]
        _draw_t(output, H, W, r, c, dr, dc, bar_dist, v)
    
    return output

def _draw_t(grid, H, W, pr, pc, dr, dc, bar_dist, color):
    # Arm: from pixel in direction (dr, dc) for bar_dist cells (including pixel)
    for i in range(bar_dist):
        r, c = pr + dr * i, pc + dc * i
        if 0 <= r < H and 0 <= c < W:
            grid[r][c] = color
    
    # Bar: perpendicular to direction, at bar_dist from pixel, ±2
    bar_r, bar_c = pr + dr * bar_dist, pc + dc * bar_dist
    if dr == 0:
        # Horizontal arm → vertical bar
        for i in range(-2, 3):
            r, c = bar_r + i, bar_c
            if 0 <= r < H and 0 <= c < W:
                grid[r][c] = color
    else:
        # Vertical arm → horizontal bar
        for i in range(-2, 3):
            r, c = bar_r, bar_c + i
            if 0 <= r < H and 0 <= c < W:
                grid[r][c] = color
    
    # Antlers: 1 past bar, at the bar ends (±2 perpendicular)
    ant_r, ant_c = pr + dr * (bar_dist + 1), pc + dc * (bar_dist + 1)
    if dr == 0:
        # Horizontal arm → antlers at ±2 rows from ant
        for i in [-2, 2]:
            r, c = ant_r + i, ant_c
            if 0 <= r < H and 0 <= c < W:
                grid[r][c] = color
    else:
        for i in [-2, 2]:
            r, c = ant_r, ant_c + i
            if 0 <= r < H and 0 <= c < W:
                grid[r][c] = color
