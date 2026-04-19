from collections import Counter, defaultdict

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    cells = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not cells:
        return grid
    
    n = len(cells)
    
    # Form clusters using Chebyshev distance <= 2 for min output size
    par = list(range(n))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    def union(x, y):
        a, b = find(x), find(y)
        if a != b: par[a] = b
    for i in range(n):
        for j in range(i+1, n):
            if max(abs(cells[i][0]-cells[j][0]), abs(cells[i][1]-cells[j][1])) <= 2:
                union(i, j)
    grps = defaultdict(list)
    for i in range(n): grps[find(i)].append(i)
    
    min_H = min_W = 1
    for cl in grps.values():
        rs = [cells[i][0] for i in cl]; cs = [cells[i][1] for i in cl]
        min_H = max(min_H, max(rs)-min(rs)+1)
        min_W = max(min_W, max(cs)-min(cs)+1)
    
    for area in range(min_H * min_W, 50):
        for H in range(min_H, min(area+1, 10)):
            if area % H != 0: continue
            W = area // H
            if W < min_W or W > 10: continue
            result = _try_size(cells, bg, H, W, grid)
            if result is not None:
                return result
    return grid

def _try_size(cells, bg, H, W, grid):
    n = len(cells)
    rows, cols = len(grid), len(grid[0])
    par = list(range(n))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    def union(x, y):
        a, b = find(x), find(y)
        if a != b: par[a] = b
    for i in range(n):
        for j in range(i+1, n):
            if abs(cells[i][0]-cells[j][0]) < H and abs(cells[i][1]-cells[j][1]) < W:
                union(i, j)
    grps = defaultdict(list)
    for i in range(n): grps[find(i)].append(i)
    
    cluster_data = []
    for cl in grps.values():
        r0, c0 = cells[cl[0]][0], cells[cl[0]][1]
        members = [(cells[i][0]-r0, cells[i][1]-c0, cells[i][2]) for i in cl]
        dr_lo = max(-m[0] for m in members)
        dr_hi = min(H-1-m[0] for m in members)
        dc_lo = max(-m[1] for m in members)
        dc_hi = min(W-1-m[1] for m in members)
        if dr_lo > dr_hi or dc_lo > dc_hi: return None
        valid = [(dr0, dc0) for dr0 in range(dr_lo, dr_hi+1) for dc0 in range(dc_lo, dc_hi+1)]
        cluster_data.append((members, valid, cl))
    
    cluster_data.sort(key=lambda x: len(x[1]))
    pattern = [[None]*W for _ in range(H)]
    all_solutions = []
    count = [0]
    
    def backtrack(ci):
        count[0] += 1
        if count[0] > 50000 or len(all_solutions) > 50: return
        if ci == len(cluster_data):
            filled = sum(1 for r in range(H) for c in range(W) if pattern[r][c] is not None)
            if filled >= max(2, (H * W + 1) // 2):
                sol = [[bg if pattern[r][c] is None else pattern[r][c] for c in range(W)] for r in range(H)]
                # Compute assignment: for each cell, find its pattern pos
                assigns = {}
                for ci2 in range(len(cluster_data)):
                    members, valid, cl_indices = cluster_data[ci2]
                    # We need to find which anchor was used - stored during backtrack
                    # Actually we can recover from pattern values
                all_solutions.append(sol)
            return
        members, valid, cl_indices = cluster_data[ci]
        for (dr0, dc0) in valid:
            ok = True; assigns = []
            for (dr, dc, v) in members:
                pr, pc = dr0+dr, dc0+dc
                if pattern[pr][pc] is not None and pattern[pr][pc] != v:
                    ok = False; break
                assigns.append((pr, pc, v))
            if not ok: continue
            old = []
            for (pr, pc, v) in assigns:
                old.append((pr, pc, pattern[pr][pc])); pattern[pr][pc] = v
            backtrack(ci+1)
            for (pr, pc, ov) in old: pattern[pr][pc] = ov
    
    backtrack(0)
    if not all_solutions: return None
    
    def score_pattern(pat):
        # Primary: minimize hidden cells
        hidden = 0
        used_origins = set()
        for r, c, v in cells:
            best_hidden = float('inf')
            for dr in range(H):
                for dc in range(W):
                    if pat[dr][dc] == v:
                        orig = (r-dr, c-dc)
                        h = 0
                        for pr in range(H):
                            for pc in range(W):
                                gr, gc = orig[0]+pr, orig[1]+pc
                                if 0 <= gr < rows and 0 <= gc < cols:
                                    if pat[pr][pc] != bg and grid[gr][gc] == bg:
                                        h += 1
                        if h < best_hidden:
                            best_hidden = h
            hidden += best_hidden
        # Secondary: prefer bg at later positions (maximize sum of bg indices)
        bg_index_sum = sum(r*W + c for r in range(H) for c in range(W) if pat[r][c] == bg)
        return (-hidden, bg_index_sum)
    
    best = max(all_solutions, key=score_pattern)
    return best
