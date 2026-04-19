from collections import Counter

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    H = len(input_grid)
    W = len(input_grid[0])
    bg = Counter(v for row in input_grid for v in row).most_common(1)[0][0]
    
    pixels = [(r, c, input_grid[r][c]) for r in range(H) for c in range(W) if input_grid[r][c] != bg]
    
    # Output starts as all 1s
    out = [[1]*W for _ in range(H)]
    
    for r0, c0, v in pixels:
        # Trace bouncing diagonal from (r0, c0)
        visited = set()
        # Try all 4 initial directions
        for dr0 in [1, -1]:
            for dc0 in [1, -1]:
                r, c = r0, c0
                dr, dc = dr0, dc0
                for _ in range(2 * (H + W)):
                    visited.add((r, c))
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= H:
                        dr = -dr
                        nr = r + dr
                    if nc < 0 or nc >= W:
                        dc = -dc
                        nc = c + dc
                    if (nr, nc) in visited:
                        break
                    r, c = nr, nc
        
        for (r, c) in visited:
            out[r][c] = v
    
    return out
