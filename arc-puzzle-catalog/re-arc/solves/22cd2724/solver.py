import copy

def transform(input_grid):
    grid = copy.deepcopy(input_grid)
    H = len(grid)
    W = len(grid[0])
    C = W - H - 1  # Center of right triangle
    
    for r in range(H):
        # Left inverted triangle: cols 0 to (H-2-r)
        left_end = H - 2 - r
        if left_end >= 0:
            for c in range(left_end + 1):
                grid[r][c] = 3
        
        # Right expanding triangle: centered at C, width 2*(r-1)+1 for r >= 1
        if r >= 1:
            left_col = C - (r - 1)
            right_col = C + (r - 1)
            for c in range(max(0, left_col), min(W, right_col + 1)):
                grid[r][c] = 3
    
    return grid
