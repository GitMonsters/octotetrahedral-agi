import copy
from collections import Counter

def transform(input_grid):
    R, C = len(input_grid), len(input_grid[0])
    grid = copy.deepcopy(input_grid)
    
    # Find bg (most common) and marker (other value)
    flat = [v for r in input_grid for v in r]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all marker positions
    markers = []
    for r in range(R):
        for c in range(C):
            if input_grid[r][c] != bg:
                markers.append((r, c))
    
    for r, c in markers:
        grid[r][c] = bg
        # Place marks at diagonal neighbors
        if r > 0 and c > 0:
            grid[r-1][c-1] = 5   # up-left
        if r > 0 and c < C-1:
            grid[r-1][c+1] = 6   # up-right
        if r < R-1 and c > 0:
            grid[r+1][c-1] = 9   # down-left
        if r < R-1 and c < C-1:
            grid[r+1][c+1] = 6   # down-right
    
    return grid
