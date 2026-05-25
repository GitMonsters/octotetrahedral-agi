import json, sys
from collections import Counter

def solve(grid):
    H = len(grid)
    W = len(grid[0])
    bg = 0
    
    color_count = Counter()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                color_count[grid[r][c]] += 1
    region_colors = [c for c, _ in color_count.most_common(2)]
    color_A, color_B = region_colors
    other = {color_A: color_B, color_B: color_A}
    
    visited = set()
    large_region = set()
    cell_color = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] == bg or (r,c) in visited: continue
            color = grid[r][c]
            comp = []
            queue = [(r,c)]
            visited.add((r,c))
            while queue:
                cr, cc = queue.pop(0)
                comp.append((cr,cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if (nr,nc) not in visited and 0<=nr<H and 0<=nc<W and grid[nr][nc] == color:
                        visited.add((nr,nc))
                        queue.append((nr,nc))
            if len(comp) >= 10:
                for cell in comp:
                    large_region.add(cell)
                    cell_color[cell] = color
    
    cleaned = set(large_region)
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for (r,c) in cleaned:
            my_color = cell_color[(r,c)]
            same_count = 0
            bg_or_oob = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if not (0<=nr<H and 0<=nc<W):
                    bg_or_oob += 1
                elif (nr,nc) in cleaned and cell_color[(nr,nc)] == my_color:
                    same_count += 1
                elif (nr,nc) not in cleaned:
                    bg_or_oob += 1
            if same_count <= 1 and bg_or_oob >= 2:
                to_remove.add((r,c))
        if to_remove:
            cleaned -= to_remove
            changed = True
    
    # One-at-a-time foreign cell detection to avoid cascade errors
    foreign_cells = {}
    changed = True
    while changed:
        changed = False
        for (r,c) in sorted(cleaned):
            if (r,c) in foreign_cells: continue
            my_color = cell_color[(r,c)]
            if my_color not in other: continue
            same_count = 0
            other_count = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr,nc) in cleaned:
                    effective = foreign_cells.get((nr,nc), cell_color[(nr,nc)])
                    if effective == my_color:
                        same_count += 1
                    elif effective in other:
                        other_count += 1
            if same_count <= 1 and other_count >= 3:
                foreign_cells[(r,c)] = other[my_color]
                changed = True
                break
    
    holes = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg: continue
            nc = Counter()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr_, nc_ = r+dr, c+dc
                if (nr_,nc_) in cleaned:
                    eff = foreign_cells.get((nr_,nc_), cell_color[(nr_,nc_)])
                    nc[eff] += 1
            for color, count in nc.most_common():
                if count >= 2 and color in other:
                    holes.append((r, c, color))
                    break
    
    out = [row[:] for row in grid]
    
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and (r,c) not in large_region:
                out[r][c] = bg
    
    for (r,c) in large_region - cleaned:
        out[r][c] = bg
    
    for (r,c), new_color in foreign_cells.items():
        out[r][c] = new_color
    
    for r, c, color in holes:
        out[r][c] = color
    
    for r, c, region_color in holes:
        fill_color = other[region_color]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and not (dr == 0 and dc == 0):
                    out[nr][nc] = fill_color
    for r, c, _ in holes:
        out[r][c] = 8
    
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
