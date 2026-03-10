import json
from collections import deque, Counter

def solve(grid):
    H, W = len(grid), len(grid[0])
    
    # 1. Detect background tile
    pc = 1
    for p in range(1, min(W//2, 16) + 1):
        if all(grid[0][c] == grid[0][c % p] for c in range(W)):
            pc = p; break
    pr = 1
    for p in range(1, min(H//2, 16) + 1):
        if all(grid[r][0] == grid[r % p][0] for r in range(H)):
            pr = p; break
    tile = [[grid[r][c] for c in range(pc)] for r in range(pr)]
    
    # 2. Find non-background cells and 4-connected components
    non_bg = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != tile[r % pr][c % pc]:
                non_bg.add((r, c))
    
    visited = set()
    components = []
    for r, c in sorted(non_bg):
        if (r, c) in visited:
            continue
        comp = set()
        queue = deque([(r, c)])
        while queue:
            cr, cc = queue.popleft()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            comp.add((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and (nr, nc) in non_bg:
                    queue.append((nr, nc))
        components.append(comp)
    
    # 3. Merge components with same border color and adjacent bounding boxes
    def comp_bbox(comp):
        rs = [r for r,c in comp]
        cs = [c for r,c in comp]
        return (min(rs), min(cs), max(rs), max(cs))
    
    def comp_border_color(comp):
        r0, c0, r1, c1 = comp_bbox(comp)
        cnt = Counter()
        for r, c in comp:
            if r == r0 or r == r1 or c == c0 or c == c1:
                cnt[grid[r][c]] += 1
        return cnt.most_common(1)[0][0] if cnt else -1
    
    n = len(components)
    bboxes = [comp_bbox(c) for c in components]
    bcols = [comp_border_color(c) for c in components]
    
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(i)] = find(j)
    
    max_gap = max(pr, pc)
    for i in range(n):
        for j in range(i+1, n):
            if bcols[i] != bcols[j]:
                continue
            r0i, c0i, r1i, c1i = bboxes[i]
            r0j, c0j, r1j, c1j = bboxes[j]
            same_rows = abs(r0i - r0j) <= 1 and abs(r1i - r1j) <= 1
            same_cols = abs(c0i - c0j) <= 1 and abs(c1i - c1j) <= 1
            col_gap = max(c0i, c0j) - min(c1i, c1j)
            row_gap = max(r0i, r0j) - min(r1i, r1j)
            if same_rows and col_gap <= max_gap:
                union(i, j)
            elif same_cols and row_gap <= max_gap:
                union(i, j)
    
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = set()
        groups[root].update(components[i])
    merged = list(groups.values())
    merged.sort(key=len, reverse=True)
    rects = merged[:2]
    
    # 4. Extract rectangle info using grid-based bbox analysis
    def extract_rect(comp):
        r0, c0, r1, c1 = comp_bbox(comp)
        perim_cnt = Counter()
        for c in range(c0, c1+1):
            perim_cnt[grid[r0][c]] += 1
            perim_cnt[grid[r1][c]] += 1
        for r in range(r0+1, r1):
            perim_cnt[grid[r][c0]] += 1
            perim_cnt[grid[r][c1]] += 1
        border_color = perim_cnt.most_common(1)[0][0]
        
        # Expand bbox to reclaim border cells that coincide with the bg tile
        while c1 + 1 < W and all(grid[r][c1 + 1] == border_color for r in range(r0, r1 + 1)):
            c1 += 1
        while c0 - 1 >= 0 and all(grid[r][c0 - 1] == border_color for r in range(r0, r1 + 1)):
            c0 -= 1
        while r1 + 1 < H and all(grid[r1 + 1][c] == border_color for c in range(c0, c1 + 1)):
            r1 += 1
        while r0 - 1 >= 0 and all(grid[r0 - 1][c] == border_color for c in range(c0, c1 + 1)):
            r0 -= 1
        
        interior = []
        for r in range(r0+1, r1):
            row = []
            for c in range(c0+1, c1):
                row.append(grid[r][c])
            interior.append(row)
        
        int_cnt = Counter()
        for row in interior:
            for v in row:
                int_cnt[v] += 1
        int_bg = int_cnt.most_common(1)[0][0]
        
        return interior, border_color, int_bg
    
    int0, bord0, bg0 = extract_rect(rects[0])
    int1, bord1, bg1 = extract_rect(rects[1])
    
    # 5. Find marks (non-interior-bg cells)
    def find_marks(interior, int_bg):
        marks = {}
        for r, row in enumerate(interior):
            for c, v in enumerate(row):
                if v != int_bg:
                    marks[(r, c)] = v
        return marks
    
    marks0 = find_marks(int0, bg0)
    marks1 = find_marks(int1, bg1)
    colors0 = set(marks0.values())
    colors1 = set(marks1.values())
    
    # Source has more unique colors; target marker colors are subset of source
    if len(colors0) >= len(colors1) and colors1.issubset(colors0):
        source_marks = marks0
        target_int, target_bg, target_bord = int1, bg1, bord1
        target_marks = marks1
    else:
        source_marks = marks1
        target_int, target_bg, target_bord = int0, bg0, bord0
        target_marks = marks0
    
    center_colors = set(target_marks.values())
    
    # 6. Cluster source marks (8-connected) to find stamp templates
    src_set = set(source_marks.keys())
    src_visited = set()
    stamp_clusters = []
    for pos in sorted(src_set):
        if pos in src_visited:
            continue
        cluster = {}
        queue = deque([pos])
        while queue:
            p = queue.popleft()
            if p in src_visited:
                continue
            src_visited.add(p)
            cluster[p] = source_marks[p]
            r, c = p
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    np = (r + dr, c + dc)
                    if np in src_set and np not in src_visited:
                        queue.append(np)
        stamp_clusters.append(cluster)
    
    # 7. Build stamp templates keyed by center color
    stamp_templates = {}
    for cluster in stamp_clusters:
        centers = [(pos, col) for pos, col in cluster.items() if col in center_colors]
        if len(centers) != 1:
            continue
        center_pos, center_col = centers[0]
        cr, cc = center_pos
        template = {}
        for (r, c), col in cluster.items():
            template[(r - cr, c - cc)] = col
        stamp_templates[center_col] = template
    
    # 8. Place stamps at target marker positions
    tH, tW = len(target_int), len(target_int[0])
    output_int = [row[:] for row in target_int]
    
    for (r, c), col in target_marks.items():
        if col in stamp_templates:
            for (dr, dc), scol in stamp_templates[col].items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < tH and 0 <= nc < tW:
                    output_int[nr][nc] = scol
    
    # 9. Add border
    output = [[target_bord] * (tW + 2)]
    for row in output_int:
        output.append([target_bord] + row + [target_bord])
    output.append([target_bord] * (tW + 2))
    
    return output


