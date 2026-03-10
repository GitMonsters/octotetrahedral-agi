import json, sys
from collections import Counter

PATTERNS = {
    2: (4, [[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]]),
    3: (1, [[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]]),
    5: (6, [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]),
    8: (7, [[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]]),
}

def solve(grid):
    H = len(grid)
    W = len(grid[0])
    bg = 0
    
    cells = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                cells[(r,c)] = grid[r][c]
    
    visited = set()
    components = []
    for (r,c) in cells:
        if (r,c) in visited: continue
        comp = []
        queue = [(r,c)]
        visited.add((r,c))
        while queue:
            cr,cc = queue.pop(0)
            comp.append((cr,cc))
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc = cr+dr,cc+dc
                if (nr,nc) not in visited and (nr,nc) in cells:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        components.append(comp)
    
    # Collect all multi-cell components as shape parts
    shape_cells = []
    dots = []
    for comp in components:
        if len(comp) > 1:
            shape_cells.extend(comp)
        else:
            dots.append(comp[0])
    
    marker = None
    if shape_cells:
        # Determine the primary shape color from multi-cell components
        colors = Counter(cells[(r,c)] for r,c in shape_cells)
        shape_color = colors.most_common(1)[0][0]

        # Include isolated cells of shape_color as part of the shape
        # (handles disconnected patterns like the X-shape)
        new_dots = []
        for d in dots:
            if cells[d] == shape_color:
                shape_cells.append(d)
            else:
                new_dots.append(d)
        dots = new_dots

        # Check for superimposed dot (different-color cell in shape)
        for (r,c) in shape_cells:
            if cells[(r,c)] != shape_color:
                dots.append((r,c))

        min_r = min(r for r,c in shape_cells)
        min_c = min(c for r,c in shape_cells)
        marker_pos = (min_r + 4, min_c + 4)

        if marker_pos in cells:
            marker = marker_pos
            dots = [d for d in dots if d != marker]
    
    out = [[bg]*W for _ in range(H)]
    
    for (r,c) in dots:
        color = cells[(r,c)]
        if color not in PATTERNS: continue
        out_color, mask = PATTERNS[color]
        for dr in range(4):
            for dc in range(4):
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and mask[dr][dc]:
                    out[nr][nc] = out_color
    
    return out

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        match = result == ex['output']
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            import numpy as np
            r = np.array(result)
            e = np.array(ex['output'])
            diffs = np.argwhere(r != e)
            print(f"  {len(diffs)} diffs")
            for d in diffs[:10]:
                print(f"  ({d[0]},{d[1]}): got {r[d[0],d[1]]}, exp {e[d[0],d[1]]}")
