from collections import Counter, deque

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    flat = [c for row in grid for c in row]
    freq = Counter(flat)
    
    bg = freq.most_common(1)[0][0]
    non_bg = sorted(freq.keys() - {bg})
    if not non_bg:
        return [row[:] for row in grid]
    shape_color = min(non_bg, key=lambda c: freq[c])
    
    # Find 8-connected components of shape color
    visited = [[False]*cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == shape_color and not visited[r][c]:
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == shape_color:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                components.append(comp)
    
    # Merge components whose expanded bboxes overlap
    def bbox(cells):
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        return (min(rs), min(cs), max(rs), max(cs))
    
    def exp_bbox(b):
        return (max(0, b[0]-1), max(0, b[1]-1), min(rows-1, b[2]+1), min(cols-1, b[3]+1))
    
    def overlaps(b1, b2):
        return b1[0] <= b2[2] and b2[0] <= b1[2] and b1[1] <= b2[3] and b2[1] <= b1[3]
    
    parent = list(range(len(components)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    changed = True
    while changed:
        changed = False
        groups = {}
        for i in range(len(components)):
            root = find(i)
            groups.setdefault(root, [])
            groups[root].append(i)
        
        roots = list(groups.keys())
        for a in range(len(roots)):
            for b in range(a + 1, len(roots)):
                ra, rb = roots[a], roots[b]
                cells_a = [c for i in groups[ra] for c in components[i]]
                cells_b = [c for i in groups[rb] for c in components[i]]
                eb_a = exp_bbox(bbox(cells_a))
                eb_b = exp_bbox(bbox(cells_b))
                if overlaps(eb_a, eb_b):
                    union(ra, rb)
                    changed = True
        if changed:
            break  # restart grouping after any merge
    
    # Rebuild final groups
    final_groups = {}
    for i in range(len(components)):
        root = find(i)
        final_groups.setdefault(root, [])
        final_groups[root].extend(components[i])
    
    # Fill background cells in each group's bbox with 2
    result = [row[:] for row in grid]
    for cells in final_groups.values():
        b = bbox(cells)
        for r in range(b[0], b[2] + 1):
            for c in range(b[1], b[3] + 1):
                if grid[r][c] == bg:
                    result[r][c] = 2
    
    return result
