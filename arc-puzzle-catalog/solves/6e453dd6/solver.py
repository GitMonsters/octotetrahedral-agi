import json, sys
from collections import deque

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    g = [row[:] for row in grid]
    bg = 6
    
    white_col = None
    for c in range(cols):
        if all(g[r][c] == 5 for r in range(rows)):
            white_col = c
            break
    
    target_right = white_col - 1
    
    visited = [[False]*cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != 0 or visited[r][c]:
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
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and g[nr][nc] == 0:
                        q.append((nr, nc))
            components.append(comp)
    
    for r in range(rows):
        for c in range(cols):
            if g[r][c] == 0:
                g[r][c] = bg
    
    for comp in components:
        max_col = max(c for r, c in comp)
        shift = target_right - max_col
        
        row_cols = {}
        for r, c in comp:
            nc = c + shift
            if 0 <= nc < cols:
                g[r][nc] = 0
                row_cols.setdefault(r, set()).add(nc)
        
        for r, col_set in row_cols.items():
            left, right = min(col_set), max(col_set)
            if right == target_right:
                has_gap = any(c not in col_set for c in range(left, right + 1))
                if has_gap:
                    for c in range(white_col + 1, cols):
                        g[r][c] = 2
    
    return g

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        print(f"Train {i}: {'PASS' if result == ex['output'] else 'FAIL'}")
