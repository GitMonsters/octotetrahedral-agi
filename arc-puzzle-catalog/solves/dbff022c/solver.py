"""Solver for dbff022c — Frame Fill via Color Key"""
import json
from typing import List
from collections import deque

def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    
    # Find connected components of non-zero cells
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r,c) not in visited:
                comp = set()
                q = deque([(r,c)])
                visited.add((r,c))
                while q:
                    cr, cc = q.popleft()
                    comp.add((cr,cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and grid[nr][nc]!=0:
                            visited.add((nr,nc))
                            q.append((nr,nc))
                components.append(comp)
    
    # Key = component with most unique colors
    key_cells = max(components, key=lambda c: len(set(grid[r][c] for r,c in c)))
    
    # Parse key
    kr = [r for r,c in key_cells]
    kc = [c for r,c in key_cells]
    r0, r1, c0, c1 = min(kr), max(kr), min(kc), max(kc)
    krows, kcols = r1-r0+1, c1-c0+1
    
    forward = {}
    reverse = {}
    if krows == 2:  # column pairs (top→bottom)
        for c in range(c0, c1+1):
            if grid[r0][c] != 0 and grid[r1][c] != 0:
                forward[grid[r0][c]] = grid[r1][c]
                reverse[grid[r1][c]] = grid[r0][c]
    elif kcols == 2:  # row pairs (left→right)
        for r in range(r0, r1+1):
            if grid[r][c0] != 0 and grid[r][c1] != 0:
                forward[grid[r][c0]] = grid[r][c1]
                reverse[grid[r][c1]] = grid[r][c0]

    # Determine correct mapping direction by checking which key set
    # has more overlap with actual frame border colors in the grid
    frame_colors = set()
    for comp in components:
        if comp is not key_cells:
            for cr, cc in comp:
                frame_colors.add(grid[cr][cc])
    fwd_hits = len(frame_colors & set(forward.keys()))
    rev_hits = len(frame_colors & set(reverse.keys()))
    mapping = reverse if rev_hits > fwd_hits else forward
    
    # Flood fill from border to find exterior cells
    exterior = set()
    q = deque()
    for r in range(rows):
        for c in range(cols):
            if (r==0 or r==rows-1 or c==0 or c==cols-1) and grid[r][c]==0:
                exterior.add((r,c))
                q.append((r,c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in exterior and grid[nr][nc]==0:
                exterior.add((nr,nc))
                q.append((nr,nc))
    
    # Interior cells
    interior = [(r,c) for r in range(rows) for c in range(cols) 
                if grid[r][c]==0 and (r,c) not in exterior]
    
    # For each interior cell, find frame color (nearest non-zero non-key cell)
    for r, c in interior:
        neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]!=0 and (nr,nc) not in key_cells:
                neighbors.append(grid[nr][nc])
        if not neighbors:
            vis = {(r,c)}
            bq = deque([(r,c)])
            while bq and not neighbors:
                br, bc = bq.popleft()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = br+dr, bc+dc
                    if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in vis:
                        vis.add((nr,nc))
                        if grid[nr][nc]!=0 and (nr,nc) not in key_cells:
                            neighbors.append(grid[nr][nc])
                        elif grid[nr][nc]==0 and (nr,nc) not in exterior:
                            bq.append((nr,nc))
        if neighbors:
            frame_color = max(set(neighbors), key=neighbors.count)
            if frame_color in mapping:
                out[r][c] = mapping[frame_color]
    
    return out

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        match = result == ex['output']
        print(f"Train {i}: {'PASS ✓' if match else 'FAIL ✗'}")
        if not match:
            diffs = [(r,c,result[r][c],ex['output'][r][c]) for r in range(len(result)) for c in range(len(result[0])) if result[r][c]!=ex['output'][r][c]]
            for r,c,g,e in diffs[:10]: print(f"  ({r},{c}): got {g} expected {e}")
