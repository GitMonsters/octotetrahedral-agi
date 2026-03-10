import json, sys
from collections import Counter, deque

def solve(grid):
    H, W = len(grid), len(grid[0])

    color_counts = Counter(grid[r][c] for r in range(H) for c in range(W))
    bg = color_counts.most_common(1)[0][0]
    non_bg_colors = [c for c in color_counts if c != bg]

    def find_components_4(color):
        visited = set()
        components = []
        for r in range(H):
            for c in range(W):
                if grid[r][c] == color and (r, c) not in visited:
                    comp = set()
                    q = deque([(r, c)])
                    visited.add((r, c))
                    while q:
                        cr, cv = q.popleft()
                        comp.add((cr, cv))
                        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                            nr, nc = cr + dr, cv + dc
                            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == color and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                    components.append(comp)
        return components

    def is_solid_block(comp):
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        h, w = max_r - min_r + 1, max_c - min_c + 1
        return h >= 2 and w >= 2 and len(comp) == h * w

    # Classify: block colors have ALL 4-connected components as solid rectangles
    color_comps = {c: find_components_4(c) for c in non_bg_colors}
    path_color = None
    block_colors = set()
    for color in non_bg_colors:
        if all(is_solid_block(comp) for comp in color_comps[color]):
            block_colors.add(color)
        else:
            path_color = color

    if path_color is None:
        return grid

    # Build blocks list and cell-to-block map
    blocks = []  # (min_r, min_c, height, width, color)
    block_map = {}
    for color in block_colors:
        for comp in color_comps[color]:
            rs = [r for r, c in comp]
            cs = [c for r, c in comp]
            min_r, min_c = min(rs), min(cs)
            h, w = max(rs) - min_r + 1, max(cs) - min_c + 1
            bi = len(blocks)
            blocks.append((min_r, min_c, h, w, color))
            for cell in comp:
                block_map[cell] = bi

    bcc = Counter(color for _, _, _, _, color in blocks)
    source_color = min(bcc, key=lambda c: bcc[c])

    N8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # Find connected components of path cells (8-connectivity)
    visited = set()
    path_components = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == path_color and (r, c) not in visited and (r, c) not in block_map:
                comp = set()
                adj_blocks = set()
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cv = q.popleft()
                    comp.add((cr, cv))
                    for dr, dc in N8:
                        nr, nc = cr + dr, cv + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if (nr, nc) in block_map:
                                adj_blocks.add(block_map[(nr, nc)])
                            elif grid[nr][nc] == path_color and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                path_components.append((comp, adj_blocks))

    # Build block graph from path components
    adj = {}
    comp_for_edge = {}
    for ci, (comp, ab) in enumerate(path_components):
        ab_list = list(ab)
        for i in range(len(ab_list)):
            for j in range(i + 1, len(ab_list)):
                bi, bj = ab_list[i], ab_list[j]
                adj.setdefault(bi, set()).add(bj)
                adj.setdefault(bj, set()).add(bi)
                key = (min(bi, bj), max(bi, bj))
                comp_for_edge[key] = ci

    # BFS shortest path between source blocks
    source_blocks = [bi for bi, (_, _, _, _, c) in enumerate(blocks) if c == source_color]
    if len(source_blocks) != 2:
        return grid
    s, t = source_blocks

    q = deque([s])
    parent = {s: None}
    while q:
        node = q.popleft()
        if node == t:
            break
        for nb in adj.get(node, set()):
            if nb not in parent:
                parent[nb] = node
                q.append(nb)

    if t not in parent:
        return grid

    path_blocks = []
    cur = t
    while cur is not None:
        path_blocks.append(cur)
        cur = parent[cur]
    path_blocks.reverse()

    output = [row[:] for row in grid]

    # Change intermediate blocks to green (3)
    for bi in path_blocks:
        if blocks[bi][4] != source_color:
            min_r, min_c, h, w, _ = blocks[bi]
            for dr in range(h):
                for dc in range(w):
                    output[min_r + dr][min_c + dc] = 3

    # Change path cells to white (5)
    for i in range(len(path_blocks) - 1):
        bi, bj = path_blocks[i], path_blocks[i + 1]
        key = (min(bi, bj), max(bi, bj))
        ci = comp_for_edge.get(key)
        if ci is not None:
            for r, c in path_components[ci][0]:
                output[r][c] = 5

    return output

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        print(f"Train {i}: {'PASS' if result == ex['output'] else 'FAIL'}")
        if result != ex['output']:
            diffs = [(r,c,result[r][c],ex['output'][r][c])
                     for r in range(len(ex['output'])) for c in range(len(ex['output'][0]))
                     if result[r][c] != ex['output'][r][c]]
            print(f"  {len(diffs)} diffs: {diffs[:10]}")
