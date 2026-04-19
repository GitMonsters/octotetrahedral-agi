from collections import Counter, deque

def get_components(grid, bg):
    H, W = len(grid), len(grid[0])
    visited = [[False]*W for _ in range(H)]
    comps = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                cells = []
                q = deque([(r,c)])
                visited[r][c] = True
                while q:
                    cr,cc = q.popleft()
                    cells.append((cr,cc))
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if dr==0 and dc==0: continue
                            nr,nc = cr+dr, cc+dc
                            if 0<=nr<H and 0<=nc<W and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                q.append((nr,nc))
                comps.append((color, set(cells)))
    return comps

def transform(grid):
    H, W = len(grid), len(grid[0])
    ic = Counter(v for row in grid for v in row)
    bg = ic.most_common(1)[0][0]
    out = [row[:] for row in grid]
    
    for iteration in range(15):
        changed = False
        comps = get_components(out, bg)
        cell_to_comp = {}
        for idx, (color, cells) in enumerate(comps):
            for r,c in cells:
                cell_to_comp[(r,c)] = idx
        
        for idx, (comp_color, comp_cells) in enumerate(comps):
            if len(comp_cells) < 2: continue
            
            adj_seeds = set()
            for r,c in comp_cells:
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr,nc = r+dr, c+dc
                        if 0<=nr<H and 0<=nc<W and out[nr][nc] != bg and out[nr][nc] != comp_color:
                            sidx = cell_to_comp.get((nr,nc))
                            if sidx is not None and len(comps[sidx][1]) == 1:
                                adj_seeds.add((nr,nc, out[nr][nc]))
            
            for sr,sc,seed_color in adj_seeds:
                rs = [r for r,c in comp_cells]; cs = [c for r,c in comp_cells]
                rmin,rmax = min(rs),max(rs); cmin,cmax = min(cs),max(cs)
                
                on_right = sc > cmax
                on_left = sc < cmin
                on_bottom = sr > rmax
                on_top = sr < rmin
                
                if on_right or on_left or on_bottom or on_top:
                    for r,c in comp_cells:
                        nr, nc = r, c
                        if on_right: nc = 2*cmax + 1 - c
                        elif on_left: nc = 2*cmin - 1 - c
                        if on_bottom: nr = 2*rmax + 1 - r
                        elif on_top: nr = 2*rmin - 1 - r
                        if 0<=nr<H and 0<=nc<W and out[nr][nc] == bg:
                            out[nr][nc] = seed_color; changed = True
                else:
                    adj_comp = [(r,c) for r,c in comp_cells if abs(r-sr)<=1 and abs(c-sc)<=1]
                    if adj_comp:
                        cr,cc = adj_comp[0]
                        for r,c in comp_cells:
                            nr = sr + (sr - r) + (sr - cr)
                            nc = sc + (sc - c) + (sc - cc)
                            if 0<=nr<H and 0<=nc<W and out[nr][nc] == bg:
                                out[nr][nc] = seed_color; changed = True
        
        if not changed: break
    return out
