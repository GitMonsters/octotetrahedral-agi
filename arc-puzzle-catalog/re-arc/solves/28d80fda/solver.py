from collections import Counter, deque


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1
    bg_color = color_counts.most_common(1)[0][0]

    # Find connected components of non-background cells (8-connected)
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def bfs(sr, sc):
        queue = deque([(sr, sc)])
        visited[sr][sc] = True
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != bg_color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
        return cells

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg_color:
                components.append(bfs(r, c))

    components.sort(key=lambda c: len(c), reverse=True)

    def get_bb(cells):
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        return min(rs), min(cs), max(rs), max(cs)

    # Largest component = big rectangle
    rect_cells = components[0]
    r1, c1, r2, c2 = get_bb(rect_cells)
    rect_h = r2 - r1 + 1
    rect_w = c2 - c1 + 1
    rect_color = Counter(grid[r][c] for r, c in rect_cells).most_common(1)[0][0]

    # Find holes (non-rect-color cells inside the rectangle bounding box)
    holes = []
    for r in range(rect_h):
        for c in range(rect_w):
            if grid[r1 + r][c1 + c] != rect_color:
                holes.append((r, c))

    # Group holes by 8-connectivity
    hole_set = set(holes)
    hole_visited = set()
    hole_groups = []
    for start in holes:
        if start in hole_visited:
            continue
        queue = deque([start])
        hole_visited.add(start)
        group = []
        while queue:
            r, c = queue.popleft()
            group.append((r, c))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in hole_set and (nr, nc) not in hole_visited:
                        hole_visited.add((nr, nc))
                        queue.append((nr, nc))
        hole_groups.append(group)

    # Extract small patterns from remaining components
    patterns = []
    for comp in components[1:]:
        pr1, pc1, pr2, pc2 = get_bb(comp)
        ph, pw = pr2 - pr1 + 1, pc2 - pc1 + 1
        pat_grid = [[grid[pr1 + r][pc1 + c] for c in range(pw)] for r in range(ph)]
        patterns.append(pat_grid)

    def get_orientations(g):
        def rotate_90(g):
            H, W = len(g), len(g[0])
            return [[g[H - 1 - nc][nr] for nc in range(H)] for nr in range(W)]

        def reflect_h(g):
            return [row[::-1] for row in g]

        orientations = []
        cur = [row[:] for row in g]
        for _ in range(4):
            orientations.append(cur)
            orientations.append(reflect_h(cur))
            cur = rotate_90(cur)
        unique, seen = [], set()
        for o in orientations:
            key = tuple(tuple(row) for row in o)
            if key not in seen:
                seen.add(key)
                unique.append(o)
        return unique

    # Output: rectangle filled with rect_color, then place special colors
    output = [[rect_color] * rect_w for _ in range(rect_h)]

    used = set()
    for group in hole_groups:
        gr1, gc1, gr2, gc2 = get_bb(group)
        gh, gw = gr2 - gr1 + 1, gc2 - gc1 + 1
        hole_mask = set((r - gr1, c - gc1) for r, c in group)

        for pi, pat in enumerate(patterns):
            if pi in used:
                continue
            for o_grid in get_orientations(pat):
                oh, ow = len(o_grid), len(o_grid[0])
                if oh != gh or ow != gw:
                    continue
                o_rect = set()
                o_special = []
                for r in range(oh):
                    for c in range(ow):
                        if o_grid[r][c] == rect_color:
                            o_rect.add((r, c))
                        else:
                            o_special.append((r, c, o_grid[r][c]))
                if o_rect == hole_mask:
                    for r, c, color in o_special:
                        output[gr1 + r][gc1 + c] = color
                    used.add(pi)
                    break
            else:
                continue
            break

    return output
