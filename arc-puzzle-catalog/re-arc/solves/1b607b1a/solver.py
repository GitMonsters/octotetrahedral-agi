def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    if rows == 0:
        return []
    cols = len(grid[0])
    if cols == 0:
        return [[] for _ in range(rows)]
    
    # Create a copy of the grid
    result = [row[:] for row in grid]
    
    # Find the background color (most common color)
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
    background = max(color_count, key=color_count.get)
    
    # Find all 2x2 blocks of non-background color
    blocks = []
    visited = [[False] * cols for _ in range(rows)]
    
    for r in range(rows - 1):
        for c in range(cols - 1):
            if visited[r][c]:
                continue
            color = grid[r][c]
            if color != background:
                # Check if it's a 2x2 block
                if (grid[r][c+1] == color and 
                    grid[r+1][c] == color and 
                    grid[r+1][c+1] == color):
                    blocks.append((r, c, color))
                    visited[r][c] = True
                    visited[r][c+1] = True
                    visited[r+1][c] = True
                    visited[r+1][c+1] = True
    
    # For each 2x2 block, add decorations around it
    # Looking at the pattern:
    # - One row above the block: position (r-1, c-1) gets 1 (or color?), (r-1, c+2) gets 4
    # - One row below the block: position (r+2, c-1) gets 3, (r+2, c+2) gets 9
    
    # Wait, let me re-examine the examples more carefully
    # In example 1, the block at (2,1)-(3,2) with color 5:
    # Output shows at row 0: col 3 has 4
    # Output shows at row 3: cols (0,1,2,3) have (3,1,1,9)
    # So: above-left diagonal gets something, below gets "3 bg bg 9"
    
    # Let me look at example 3 which is simpler:
    # Block at (1,5)-(2,6) with color 1
    # Row 0, col 4 gets 1, col 7 gets 4
    # Row 3, col 4 gets 3, col 7 gets 9
    
    # So the pattern is:
    # Row above block (r-1): col c-1 gets 1, col c+2 gets 4
    # Row below block (r+2): col c-1 gets 3, col c+2 gets 9
    
    # But wait, the block color in example 3 is 1, and the marker above-left is also 1
    # In example 1, block color is 5, and looking at (2,1) block:
    # Row 4 (which is r+2 where r=2): positions should be col 0 and col 3
    # Output row 4: 3 1 1 9 ... Yes! col 0 is 3, col 3 is 9
    
    # But row 0 (r-1 where r=1 for block starting at row 1... wait the block is at rows 2-3)
    # Block at rows 2-3, cols 1-2
    # r-1 = 1, c-1 = 0, c+2 = 3
    # Looking at output row 1: col 3 has 4... yes!
    # But what about row 0 or something getting 1?
    # Output row 0 has all 1s (which is background)
    
    # Hmm, let me reconsider. The above decoration might be at r-1 only when possible
    # And the 1 and 4 are placed at c-1 and c+2
    
    for (r, c, color) in blocks:
        # Add decorations
        # Above the block (row r-1): put 1 at c-1, put 4 at c+2
        if r - 1 >= 0:
            if c - 1 >= 0:
                result[r-1][c-1] = 1
            if c + 2 < cols:
                result[r-1][c+2] = 4
        
        # Below the block (row r+2): put 3 at c-1, put 9 at c+2
        if r + 2 < rows:
            if c - 1 >= 0:
                result[r+2][c-1] = 3
            if c + 2 < cols:
                result[r+2][c+2] = 9
    
    return result