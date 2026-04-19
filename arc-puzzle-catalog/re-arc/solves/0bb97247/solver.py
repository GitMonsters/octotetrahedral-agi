from collections import Counter

def transform(input_grid):
    R, C = len(input_grid), len(input_grid[0])
    bg = Counter(v for r in input_grid for v in r).most_common(1)[0][0]
    
    # Find connected components of color 9
    cells9 = set()
    for r in range(R):
        for c in range(C):
            if input_grid[r][c] == 9:
                cells9.add((r, c))
    
    visited = set()
    num_components = 0
    for r, c in cells9:
        if (r, c) not in visited:
            num_components += 1
            queue = [(r, c)]
            visited.add((r, c))
            while queue:
                cr, cc = queue.pop(0)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if (nr, nc) in cells9 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
    
    # Fill order for 9s in the 3x3 output (checkerboard positions first)
    fill_order = [(2,0), (2,2), (1,1), (0,0), (0,2),
                  (2,1), (1,0), (1,2), (0,1)]
    
    out = [[bg]*3 for _ in range(3)]
    for idx in range(min(num_components, 9)):
        r, c = fill_order[idx]
        out[r][c] = 9
    
    return out
