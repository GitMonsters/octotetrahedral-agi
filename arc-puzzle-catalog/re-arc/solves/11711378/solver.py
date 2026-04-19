from collections import Counter, deque

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    H = len(input_grid)
    W = len(input_grid[0])
    # Determine bg dynamically
    all_vals = Counter(v for row in input_grid for v in row)
    bg = all_vals.most_common(1)[0][0]
    
    out = [row[:] for row in input_grid]
    
    fours = [(r,c) for r in range(H) for c in range(W) if input_grid[r][c] == 4]
    nines = [(r,c) for r in range(H) for c in range(W) if input_grid[r][c] == 9]
    
    if not fours and not nines:
        return out
    
    def get_component(sr, sc):
        visited = set()
        q = deque([(sr, sc)])
        visited.add((sr, sc))
        while q:
            r, c = q.popleft()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r+dr, c+dc
                    if 0<=nr<H and 0<=nc<W and (nr,nc) not in visited and input_grid[nr][nc] != bg:
                        visited.add((nr,nc))
                        q.append((nr,nc))
        return visited
    
    # Find source 4 (largest component among 4s)
    source_4 = None
    template_4 = []
    source_4_comp = set()
    max_size = 0
    for r, c in fours:
        comp = get_component(r, c)
        if len(comp) > max_size:
            max_size = len(comp)
            source_4 = (r, c)
            source_4_comp = comp
            template_4 = [(pr-r, pc-c, input_grid[pr][pc]) for pr,pc in comp if (pr,pc) != (r,c)]
    
    # Find source 9 (largest component among 9s not in source_4_comp)
    source_9 = None
    template_9 = []
    source_9_comp = set()
    max_size_9 = 0
    for r, c in nines:
        if (r, c) in source_4_comp:
            continue
        comp = get_component(r, c)
        if len(comp) > max_size_9:
            max_size_9 = len(comp)
            comp_nines = [(cr,cc) for cr,cc in comp if input_grid[cr][cc] == 9]
            best_center = comp_nines[0] if comp_nines else (r,c)
            best_dist = float('inf')
            for cr, cc in comp_nines:
                total = sum(abs(cr-pr)+abs(cc-pc) for pr,pc in comp)
                if total < best_dist:
                    best_dist = total
                    best_center = (cr, cc)
            source_9 = best_center
            source_9_comp = comp
            template_9 = [(pr-best_center[0], pc-best_center[1], input_grid[pr][pc]) 
                         for pr,pc in comp if (pr,pc) != best_center]
    
    # Stamp template_4 (same orientation) at non-source 4s
    if template_4:
        for r, c in fours:
            if (r, c) == source_4:
                continue
            for dr, dc, v in template_4:
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and out[nr][nc] == bg:
                    out[nr][nc] = v
    
    # Stamp template_9 (reflected dr) at non-source, non-cluster 9s
    if template_9:
        for r, c in nines:
            if (r, c) in source_4_comp or (r, c) in source_9_comp:
                continue
            for dr, dc, v in template_9:
                nr, nc = r+(-dr), c+dc
                if 0<=nr<H and 0<=nc<W and out[nr][nc] == bg:
                    out[nr][nc] = v
    
    return out
