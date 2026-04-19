from collections import Counter

def transform(grid):
    """
    For each marker (non-background cell), place a diagonal pattern:
    - Upper-left (-1,-1): yellow (4)
    - Upper-right (-1,+1): green (3)
    - Lower-left (+1,-1): magenta (6)
    - Lower-right (+1,+1): yellow (4)
    The entire output starts as background color.
    """
    R, C = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]
    out = [[bg for _ in range(C)] for _ in range(R)]
    
    markers = [(r, c) for r in range(R) for c in range(C) if grid[r][c] != bg]
    
    for r, c in markers:
        offsets = [(-1, -1, 4), (-1, 1, 3), (1, -1, 6), (1, 1, 4)]
        for dr, dc, color in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                out[nr][nc] = color
    
    return out
