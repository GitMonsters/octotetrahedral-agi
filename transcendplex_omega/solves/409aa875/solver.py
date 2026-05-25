import json, sys
from collections import Counter, deque

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    g = [row[:] for row in grid]
    
    bg = Counter(g[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    # Find connected components of non-bg cells using 8-connectivity
    non_bg = {(r,c) for r in range(rows) for c in range(cols) if g[r][c] != bg}
    visited = set()
    components = []
    for rc in non_bg:
        if rc in visited: continue
        comp = set()
        q = deque([rc])
        while q:
            cr, cc = q.popleft()
            if (cr,cc) in visited: continue
            visited.add((cr,cc))
            comp.add((cr,cc))
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = cr+dr, cc+dc
                    if (nr,nc) in non_bg and (nr,nc) not in visited:
                        q.append((nr,nc))
        components.append(comp)
    
    distance = 5
    dot_map = {}
    
    for comp in components:
        if len(comp) != 3: continue
        min_r = min(r for r,c in comp)
        max_r = max(r for r,c in comp)
        min_c = min(c for r,c in comp)
        max_c = max(c for r,c in comp)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        norm = frozenset((r-min_r, c-min_c) for r,c in comp)
        
        tip_r, tip_c, dr, dc = None, None, 0, 0
        
        if h == 2 and w == 2:
            # L-triomino
            all_corners = {(0,0),(0,1),(1,0),(1,1)}
            missing = all_corners - norm
            if len(missing) == 1:
                mr, mc = missing.pop()
                tip_r = min_r + (1 - mr)
                tip_c = min_c + (1 - mc)
                dr = -1 if mr == 1 else 1
                dc = -1 if mc == 1 else 1
        
        elif h == 2 and w == 3:
            if norm == frozenset([(0,1),(1,0),(1,2)]):
                tip_r, tip_c = min_r, min_c + 1
                dr, dc = -1, 0
            elif norm == frozenset([(0,0),(0,2),(1,1)]):
                tip_r, tip_c = max_r, min_c + 1
                dr, dc = 1, 0
        
        elif h == 3 and w == 2:
            if norm == frozenset([(0,0),(1,1),(2,0)]):
                tip_r, tip_c = min_r + 1, max_c
                dr, dc = 0, 1
            elif norm == frozenset([(0,1),(1,0),(2,1)]):
                tip_r, tip_c = min_r + 1, min_c
                dr, dc = 0, -1
        
        if tip_r is not None:
            dot_r = tip_r + distance * dr
            dot_c = tip_c + distance * dc
            if 0 <= dot_r < rows and 0 <= dot_c < cols:
                dot_map[(dot_r, dot_c)] = dot_map.get((dot_r, dot_c), 0) + 1
    
    cell_to_comp = {}
    for i, comp in enumerate(components):
        for rc in comp:
            cell_to_comp[rc] = i
    
    for (dot_r, dot_c), count in dot_map.items():
        if count > 1:
            g[dot_r][dot_c] = 1  # blue for collision
        else:
            if g[dot_r][dot_c] == bg:
                g[dot_r][dot_c] = 9
            else:
                ci = cell_to_comp.get((dot_r, dot_c))
                if ci is not None:
                    for r, c in components[ci]:
                        g[r][c] = 9
    
    return g

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        task = json.load(f)

    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        expected = ex['output']
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            diffs = [(r,c,result[r][c],expected[r][c]) for r in range(len(expected)) for c in range(len(expected[0])) if result[r][c] != expected[r][c]]
            for d in diffs[:10]:
                print(f"  ({d[0]},{d[1]}): got {d[2]}, expected {d[3]}")
