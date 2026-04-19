"""Solver for 4341642d: Pipe/bracket shapes are recolored based on topology.
- Branch point (T-shape) → 0 (black)
- 2 bends (U-shape) → 7 (orange)
- 1 bend (L-shape) → 1 (blue)"""

from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    cnt = Counter(c for row in grid for c in row)
    bg = cnt.most_common(1)[0][0]

    visited = [[False]*W for _ in range(H)]

    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                stack = [(r, c)]; visited[r][c] = True; cells = []
                while stack:
                    cr, cc = stack.pop(); cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True; stack.append((nr, nc))

                cell_set = set(cells)
                nbr = {}
                has_branch = False
                for cr, cc in cells:
                    n = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (nr, nc) in cell_set:
                            n.append((nr, nc))
                    nbr[(cr, cc)] = n
                    if len(n) >= 3:
                        has_branch = True

                if has_branch:
                    oc = 0  # T-shape / branch
                else:
                    endpoints = [c for c in cells if len(nbr[c]) == 1]
                    if len(endpoints) >= 2:
                        path = [endpoints[0]]
                        vis = {endpoints[0]}
                        while True:
                            cur = path[-1]
                            found = False
                            for n in nbr[cur]:
                                if n not in vis:
                                    path.append(n)
                                    vis.add(n)
                                    found = True
                                    break
                            if not found:
                                break
                        bends = 0
                        for i in range(1, len(path)-1):
                            pr, pc = path[i-1]
                            cr, cc = path[i]
                            nr, nc = path[i+1]
                            if (cr-pr, cc-pc) != (nr-cr, nc-cc):
                                bends += 1
                        oc = 7 if bends >= 2 else 1
                    else:
                        oc = 1

                for cr, cc in cells:
                    out[cr][cc] = oc

    return out
