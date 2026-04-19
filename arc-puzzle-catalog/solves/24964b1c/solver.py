from collections import Counter

def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    
    # Detect background as most common color
    flat = [grid[r][c] for r in range(H) for c in range(W)]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Deep copy output
    out = [row[:] for row in grid]
    
    # Find connected components of non-bg cells using 8-connectivity
    visited = [[False]*W for _ in range(H)]
    objects = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                queue = [(r, c)]
                visited[r][c] = True
                component = [(r, c)]
                while queue:
                    cr, cc = queue.pop(0)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != bg:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                                component.append((nr, nc))
                objects.append(component)
    
    for obj in objects:
        rows = [r for r, c in obj]
        cols = [c for r, c in obj]
        min_r, min_c = min(rows), min(cols)
        max_r, max_c = max(rows), max(cols)
        
        # Read the 2x2 region (some cells may be background)
        tl = grid[min_r][min_c]
        tr = grid[min_r][max_c]
        bl = grid[max_r][min_c]
        br = grid[max_r][max_c]
        
        # Place 2x2 solid blocks at diagonal offsets
        # Each block's color = diagonally opposite cell in original 2x2
        expansions = [
            (min_r - 2, min_c - 2, br),  # top-left gets bottom-right color
            (min_r - 2, min_c + 2, bl),  # top-right gets bottom-left color
            (max_r + 1, min_c - 2, tr),  # bottom-left gets top-right color
            (max_r + 1, min_c + 2, tl),  # bottom-right gets top-left color
        ]
        
        for er, ec, color in expansions:
            if color == bg:
                continue
            for dr in range(2):
                for dc in range(2):
                    nr, nc = er + dr, ec + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = color
    
    return out
