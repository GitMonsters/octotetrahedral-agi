from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background (most common color)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find the largest solid rectangle of a single non-background color
    best_rect = None
    best_area = 0
    
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] == bg:
                continue
            color = grid[r1][c1]
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    # Check if rectangle (r1, c1) to (r2, c2) is solid with 'color'
                    solid = True
                    for r in range(r1, r2 + 1):
                        for c in range(c1, c2 + 1):
                            if grid[r][c] != color:
                                solid = False
                                break
                        if not solid:
                            break
                    if solid:
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area > best_area:
                            best_area = area
                            best_rect = (r1, c1, r2, c2, color)
    
    # Create output: background everywhere, then draw the rectangle
    output = [[bg] * cols for _ in range(rows)]
    if best_rect:
        r1, c1, r2, c2, color = best_rect
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                output[r][c] = color
    
    return output
