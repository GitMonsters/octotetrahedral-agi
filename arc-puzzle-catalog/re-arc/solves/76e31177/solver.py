from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    # Find background color
    cnt = Counter()
    for r in grid:
        for c in r:
            cnt[c] += 1
    bg = cnt.most_common(1)[0][0]
    
    result = [row[:] for row in grid]
    
    # For each non-bg pixel, extend downward with alternating pattern
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color = grid[r][c]
                for dr in range(rows - r):
                    nr = r + dr
                    if dr % 2 == 0:
                        result[nr][c] = color
                    else:
                        result[nr][c] = 0
    
    return result
