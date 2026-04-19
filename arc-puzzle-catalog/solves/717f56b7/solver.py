from collections import Counter, deque, defaultdict

def transform(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    colors = Counter(v for row in grid for v in row)
    bg = colors.most_common(1)[0][0]

    visited = [[False]*W for _ in range(H)]
    all_clusters = []

    for r0 in range(H):
        for c0 in range(W):
            if visited[r0][c0] or grid[r0][c0] == bg: continue
            v = grid[r0][c0]
            q = deque([(r0,c0)]); visited[r0][c0] = True; pixels = [(r0,c0)]
            while q:
                cr2,cc2 = q.popleft()
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc = cr2+dr,cc2+dc
                    if 0<=nr<H and 0<=nc<W and not visited[nr][nc] and grid[nr][nc]==v:
                        visited[nr][nc] = True; q.append((nr,nc)); pixels.append((nr,nc))
            rows=[p[0] for p in pixels]; cols=[p[1] for p in pixels]
            all_clusters.append({'pixels': pixels, 'color': v,
                                  'r1': min(rows), 'r2': max(rows),
                                  'c1': min(cols), 'c2': max(cols)})

    template_center = None
    template_color = None
    template_pixel_set = None
    template_type = None

    for r in range(1, H-1):
        for c in range(1, W-1):
            if grid[r][c] != bg: continue
            arms = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
            arm_colors = [grid[ar][ac] for ar,ac in arms if 0<=ar<H and 0<=ac<W]
            if len(arm_colors)==4 and len(set(arm_colors))==1 and arm_colors[0]!=bg:
                template_center = (r, c)
                template_color = arm_colors[0]
                template_pixel_set = set(arms)
                template_type = 'hollow'
                break
        if template_center: break

    if template_center is None:
        for cl in all_clusters:
            v = cl['color']
            for pr, pc in cl['pixels']:
                arms = [(pr-1,pc),(pr+1,pc),(pr,pc-1),(pr,pc+1)]
                if not all(0<=ar<H and 0<=ac<W and grid[ar][ac]==v for ar,ac in arms): continue
                diags = [(pr-1,pc-1),(pr-1,pc+1),(pr+1,pc-1),(pr+1,pc+1)]
                if any(0<=dr<H and 0<=dc<W and grid[dr][dc]==v for dr,dc in diags): continue
                template_center = (pr, pc)
                template_color = v
                template_pixel_set = {(ar,ac) for ar,ac in arms} | {(pr,pc)}
                template_type = 'solid'
                break
            if template_center: break

    if template_center is None:
        return out

    cr, cc = template_center

    pointer_pixels = set()
    candidates = []
    for cl in all_clusters:
        if len(cl['pixels']) != 1: continue
        pr, pc = cl['pixels'][0]
        if (pr, pc) in template_pixel_set: continue
        if any(abs(pr-tr)+abs(pc-tc)==1 for tr,tc in template_pixel_set):
            pointer_pixels.add((pr, pc))
            candidates.append((pr, pc, cl['color']))

    if not candidates:
        return out

    if len(candidates) == 1:
        pr, pc, _ = candidates[0]
        pointer_dc = 1 if pc > cc else -1
    else:
        if template_type == 'solid':
            below = [(r,c,col) for r,c,col in candidates if r > cr]
            pr, pc, _ = below[0] if below else candidates[0]
        else:
            right = [(r,c,col) for r,c,col in candidates if c > cc]
            pr, pc, _ = right[0] if right else candidates[0]
        pointer_dc = 1 if pc > cc else -1

    target_clusters = []
    for cl in all_clusters:
        pset = set(map(tuple, cl['pixels']))
        if pset & template_pixel_set: continue
        if pset & pointer_pixels: continue
        target_clusters.append(cl)

    def paint_cross(sr, sc, clH, clW, solid):
        for ar, ac in [(sr-clH,sc),(sr+clH,sc),(sr,sc-clW),(sr,sc+clW)]:
            for dr in range(clH):
                for dc in range(clW):
                    nr, nc = ar+dr, ac+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = template_color
        if solid:
            for dr in range(clH):
                for dc in range(clW):
                    nr, nc = sr+dr, sc+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = template_color

    pointer_colors = set(col for _,_,col in candidates)

    if len(pointer_colors) >= 2:
        # Two pointer colors: pair same-size clusters of different colors.
        # Shadow center = midpoint of paired top-left corners; paint solid cross.
        colorA, colorB = sorted(pointer_colors)

        groupA = defaultdict(list)
        groupB = defaultdict(list)
        other_clusters = []

        for cl in target_clusters:
            clH = cl['r2'] - cl['r1'] + 1
            clW = cl['c2'] - cl['c1'] + 1
            if cl['color'] == colorA:
                groupA[(clH, clW)].append(cl)
            elif cl['color'] == colorB:
                groupB[(clH, clW)].append(cl)
            else:
                other_clusters.append(cl)

        for key in groupA: groupA[key].sort(key=lambda x: (x['r1'], x['c1']))
        for key in groupB: groupB[key].sort(key=lambda x: (x['r1'], x['c1']))

        for size in set(groupA.keys()) | set(groupB.keys()):
            clH, clW = size
            listA = groupA.get(size, [])
            listB = groupB.get(size, [])
            n = min(len(listA), len(listB))
            for i in range(n):
                sr = (listA[i]['r1'] + listB[i]['r1']) // 2
                sc = (listA[i]['c1'] + listB[i]['c1']) // 2
                paint_cross(sr, sc, clH, clW, True)
            for cl in listA[n:] + listB[n:]:
                center_row = (cl['r1'] + cl['r2']) / 2
                sign = 1 if center_row < cr else -1
                paint_cross(cl['r1'] + sign*clH, cl['c1'] + sign*pointer_dc*clW, clH, clW, False)

        for cl in other_clusters:
            clH = cl['r2'] - cl['r1'] + 1
            clW = cl['c2'] - cl['c1'] + 1
            center_row = (cl['r1'] + cl['r2']) / 2
            sign = 1 if center_row < cr else -1
            paint_cross(cl['r1'] + sign*clH, cl['c1'] + sign*pointer_dc*clW, clH, clW, False)
    else:
        # Single pointer color: each cluster casts its own hollow cross shadow.
        for cl in target_clusters:
            r1, r2, c1, c2 = cl['r1'], cl['r2'], cl['c1'], cl['c2']
            clH = r2 - r1 + 1
            clW = c2 - c1 + 1
            center_row = (r1 + r2) / 2
            sign = 1 if center_row < cr else -1
            paint_cross(r1 + sign*clH, c1 + sign*pointer_dc*clW, clH, clW, False)

    return out
