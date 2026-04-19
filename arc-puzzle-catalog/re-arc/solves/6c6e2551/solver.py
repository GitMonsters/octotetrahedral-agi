from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    flat = sum(grid, [])
    bg = Counter(flat).most_common(1)[0][0]
    
    pixels = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    output = [[bg] * cols for _ in range(rows)]
    
    def draw_arrow(r0, c0, dr, dc, color):
        """Draw arrow from (r0,c0) in direction (dr,dc). dr/dc is +1 or -1, one must be 0."""
        cells = []
        if dc != 0:  # horizontal arrow
            for i in range(4):
                cells.append((r0, c0 + i * dc))
            tip_c = c0 + 3 * dc
            for j in [-2, -1, 1, 2]:
                cells.append((r0 + j, tip_c))
            cells.append((r0 - 2, tip_c + dc))
            cells.append((r0 + 2, tip_c + dc))
        else:  # vertical arrow
            for i in range(4):
                cells.append((r0 + i * dr, c0))
            tip_r = r0 + 3 * dr
            for j in [-2, -1, 1, 2]:
                cells.append((tip_r, c0 + j))
            cells.append((tip_r + dr, c0 - 2))
            cells.append((tip_r + dr, c0 + 2))
        
        for r, c in cells:
            if 0 <= r < rows and 0 <= c < cols:
                output[r][c] = color
    
    if len(pixels) == 2:
        (r1, c1, v1), (r2, c2, v2) = pixels
        if r1 == r2:  # same row - horizontal
            if c1 < c2:
                draw_arrow(r1, c1, 0, 1, v1)
                draw_arrow(r2, c2, 0, -1, v2)
            else:
                draw_arrow(r1, c1, 0, -1, v1)
                draw_arrow(r2, c2, 0, 1, v2)
        elif c1 == c2:  # same column - vertical
            if r1 < r2:
                draw_arrow(r1, c1, 1, 0, v1)
                draw_arrow(r2, c2, -1, 0, v2)
            else:
                draw_arrow(r1, c1, -1, 0, v1)
                draw_arrow(r2, c2, 1, 0, v2)
    elif len(pixels) == 1:
        r0, c0, v = pixels[0]
        dists = {'left': c0, 'right': cols - 1 - c0, 'top': r0, 'bottom': rows - 1 - r0}
        nearest = min(dists, key=dists.get)
        dirs = {'left': (0, 1), 'right': (0, -1), 'top': (1, 0), 'bottom': (-1, 0)}
        dr, dc = dirs[nearest]
        draw_arrow(r0, c0, dr, dc, v)
    
    return output
