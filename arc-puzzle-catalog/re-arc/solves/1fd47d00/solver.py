def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Create a copy of the grid
    result = [row[:] for row in grid]
    
    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    background = Counter(flat).most_common(1)[0][0]
    
    # Find all 2x2 blocks of non-background colors
    blocks = []  # (row, col, color) - top-left corner of 2x2 block
    visited = set()
    
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (r, c) in visited:
                continue
            color = grid[r][c]
            if color != background:
                # Check if it's a 2x2 block
                if (grid[r][c+1] == color and 
                    grid[r+1][c] == color and 
                    grid[r+1][c+1] == color):
                    blocks.append((r, c, color))
                    visited.add((r, c))
                    visited.add((r, c+1))
                    visited.add((r+1, c))
                    visited.add((r+1, c+1))
    
    # For each block, draw diagonal lines
    for r, c, color in blocks:
        if color == 5:
            # Draw diagonal going down-right from bottom-right corner of block
            start_r, start_c = r + 2, c + 2
            dr, dc = 1, 1
        else:
            # Draw diagonal going up-left from top-left corner of block
            start_r, start_c = r - 1, c - 1
            dr, dc = -1, -1
        
        # Draw the diagonal line
        curr_r, curr_c = start_r, start_c
        while 0 <= curr_r < rows and 0 <= curr_c < cols:
            if result[curr_r][curr_c] == background:
                result[curr_r][curr_c] = color
            curr_r += dr
            curr_c += dc
    
    return result