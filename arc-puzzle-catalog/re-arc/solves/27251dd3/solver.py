from collections import Counter

def transform(grid):
    """
    2x2 blocks draw diagonal lines based on color:
    - Color 5 (gray) -> diagonal NE (up-right)
    - Color 7 (orange) -> diagonal SW (down-left)
    Lines only fill background cells.
    """
    R, C = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]
    out = [row[:] for row in grid]
    
    # Find 2x2 uniform non-bg blocks
    blocks = []
    for r in range(R - 1):
        for c in range(C - 1):
            v = grid[r][c]
            if v == bg:
                continue
            if grid[r][c+1] == v and grid[r+1][c] == v and grid[r+1][c+1] == v:
                blocks.append((r, c, v))
    
    for r, c, v in blocks:
        if v == 5:  # Gray -> diagonal NE
            i, j = r - 1, c + 2
            while 0 <= i < R and 0 <= j < C:
                if grid[i][j] == bg:
                    out[i][j] = v
                i -= 1
                j += 1
        elif v == 7:  # Orange -> diagonal SW
            i, j = r + 2, c - 1
            while 0 <= i < R and 0 <= j < C:
                if grid[i][j] == bg:
                    out[i][j] = v
                i += 1
                j -= 1
    
    return out
