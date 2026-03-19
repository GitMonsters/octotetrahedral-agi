"""
ARC Puzzle 223b4e32 Solver

Pattern: Find solid rectangles of non-background color and create a mesh pattern.
- Keep first and last rows/columns solid (edges)
- Punch holes at interior positions where both local row and column are odd
- Holes are filled with background color
"""

def transform(grid):
    grid = [row[:] for row in grid]  # Deep copy
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the background color (most common color)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            color_counts[color] = color_counts.get(color, 0) + 1
    bg_color = max(color_counts, key=color_counts.get)
    
    # Find all connected blobs using flood fill
    visited = [[False]*cols for _ in range(rows)]
    
    def find_blob(start_r, start_c, color):
        cells = []
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] != color:
                continue
            visited[r][c] = True
            cells.append((r, c))
            stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
        return cells
    
    # Find all rectangles
    rectangles = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg_color:
                color = grid[r][c]
                cells = find_blob(r, c, color)
                if cells:
                    min_r = min(cell[0] for cell in cells)
                    max_r = max(cell[0] for cell in cells)
                    min_c = min(cell[1] for cell in cells)
                    max_c = max(cell[1] for cell in cells)
                    rectangles.append((min_r, max_r, min_c, max_c, color))
    
    # For each rectangle, apply the mesh pattern
    for min_r, max_r, min_c, max_c, color in rectangles:
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                local_r = r - min_r
                local_c = c - min_c
                
                # Skip edges (first and last rows/columns stay solid)
                if local_r == 0 or local_r == height - 1:
                    continue
                if local_c == 0 or local_c == width - 1:
                    continue
                
                # Punch hole at odd row + odd col positions
                if local_r % 2 == 1 and local_c % 2 == 1:
                    grid[r][c] = bg_color
    
    return grid
