import json, sys
from collections import deque

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    g = [row[:] for row in grid]
    bg = 7
    
    key_colors = set()
    for r in range(3):
        for c in range(3):
            v = g[r][c]
            if v != bg and v != 3:
                key_colors.add(v)
    
    visited = [[False]*cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if g[r][c] == bg or visited[r][c]:
                continue
            comp = []
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                if visited[cr][cc]:
                    continue
                visited[cr][cc] = True
                comp.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and g[nr][nc] != bg:
                        q.append((nr, nc))
            components.append(comp)
    
    for comp in components:
        comp_colors = set(g[r][c] for r,c in comp if g[r][c] != 0 and g[r][c] != bg)
        if 3 in comp_colors:
            continue
        if key_colors.issubset(comp_colors):
            for r, c in comp:
                g[r][c] = bg
    
    return g

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        print(f"Train {i}: {'PASS' if result == ex['output'] else 'FAIL'}")
