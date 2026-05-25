from collections import Counter, deque

def solve(grid):
    H, W = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    
    # bg = most common, border = second most common
    all_colors = Counter(grid[r][c] for r in range(H) for c in range(W))
    sorted_colors = all_colors.most_common()
    bg = sorted_colors[0][0]
    border = sorted_colors[1][0]
    colored_colors = set(all_colors.keys()) - {bg, border}
    
    # Find junction center: border cell where entire 3x3 is border
    junction_center = None
    for r in range(1, H-1):
        for c in range(1, W-1):
            if grid[r][c] == border and all(
                grid[r+dr][c+dc] == border 
                for dr in [-1,0,1] for dc in [-1,0,1]):
                junction_center = (r, c)
                break
        if junction_center:
            break
    
    # Ring cells = 8 neighbors of center
    ring = set()
    if junction_center:
        jr, jc = junction_center
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr != 0 or dc != 0:
                    ring.add((jr+dr, jc+dc))
    
    # Find colored regions
    visited = [[False]*W for _ in range(H)]
    regions = []
    for r in range(H):
        for c in range(W):
            if not visited[r][c] and grid[r][c] in colored_colors:
                q = deque([(r,c)])
                visited[r][c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc, grid[cr][cc]))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and not visited[nr][nc] and grid[nr][nc] in colored_colors:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                regions.append(cells)
    
    region_majorities = []
    for cells in regions:
        counts = Counter(v for _,_,v in cells)
        region_majorities.append(counts.most_common(1)[0][0])
    
    
    # Multi-source BFS from colored regions through border cells
    # Ring cells act as barriers and are excluded from BFS
    dist = [[float('inf')]*W for _ in range(H)]
    color_assign = [[None]*W for _ in range(H)]
    conflict = [[False]*W for _ in range(H)]
    
    blocked = ring | ({junction_center} if junction_center else set())
    
    queue = deque()
    for ri, cells in enumerate(regions):
        maj = region_majorities[ri]
        for r, c, v in cells:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and grid[nr][nc] == border and (nr, nc) not in blocked:
                    if dist[nr][nc] > 0:
                        dist[nr][nc] = 0
                        color_assign[nr][nc] = maj
                        queue.append((nr, nc))
                    elif dist[nr][nc] == 0 and color_assign[nr][nc] != maj:
                        conflict[nr][nc] = True
    
    while queue:
        r, c = queue.popleft()
        if conflict[r][c]:
            continue
        cur_color = color_assign[r][c]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and grid[nr][nc] == border and (nr, nc) not in blocked:
                new_dist = dist[r][c] + 1
                if new_dist < dist[nr][nc]:
                    dist[nr][nc] = new_dist
                    color_assign[nr][nc] = cur_color
                    queue.append((nr, nc))
                elif new_dist == dist[nr][nc] and color_assign[nr][nc] != cur_color:
                    conflict[nr][nc] = True
    
    # Junction center color = majority of colors assigned to border cells adjacent to ring
    junction_color = None
    if junction_center:
        adj_colors = []
        for rr, rc in ring:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = rr+dr, rc+dc
                if (nr, nc) not in blocked and 0<=nr<H and 0<=nc<W and color_assign[nr][nc] is not None:
                    adj_colors.append(color_assign[nr][nc])
        if adj_colors:
            junction_color = Counter(adj_colors).most_common(1)[0][0]
    
    # Apply colors to border cells
    for r in range(H):
        for c in range(W):
            if grid[r][c] == border:
                if (r, c) in ring:
                    pass  # Ring stays as border
                elif junction_center and (r, c) == junction_center:
                    if junction_color is not None:
                        output[r][c] = junction_color
                elif not conflict[r][c] and color_assign[r][c] is not None:
                    output[r][c] = color_assign[r][c]
    
    return output


if __name__ == '__main__':
    import json, sys
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        print(f"Train {i}: {'PASS' if result == ex['output'] else 'FAIL'}")
        if result != ex['output']:
            diffs = [(r,c,result[r][c],ex['output'][r][c]) 
                     for r in range(len(ex['output'])) for c in range(len(ex['output'][0]))
                     if result[r][c] != ex['output'][r][c]]
            print(f"  {len(diffs)} diffs: {diffs[:5]}")
