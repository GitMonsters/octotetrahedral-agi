from collections import deque


def find_all_8conn_components(grid, val):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    components = []
    
    for r in range(H):
        for c in range(W):
            if grid[r][c] == val and not visited[r][c]:
                component = set()
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.add((cr, cc))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == val:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                components.append(component)
    return components


def find_enclosed_4conn(grid, bg):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    queue = deque()
    
    for r in range(H):
        for c in [0, W - 1]:
            if grid[r][c] == bg and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    for c in range(W):
        for r in [0, H - 1]:
            if grid[r][c] == bg and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == bg and not visited[nr][nc]:
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    enclosed = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] == bg and not visited[r][c]:
                enclosed.add((r, c))
    
    return enclosed


def transform(grid):
    """
    Transformation rule for puzzle 29351ece:
    
    Fill enclosed background cells when there is a significant secondary
    8-connected foreground component (>= 5% of total fg cells).
    
    The fill expands from enclosed cells that are 4-adjacent to the main
    8-connected component, including 8-adjacent cells that are also
    4-adjacent to either main or already-filled cells.
    """
    grid = [list(row) for row in grid]
    H, W = len(grid), len(grid[0])
    
    vals = sorted(set(v for row in grid for v in row))
    if len(vals) != 2:
        return grid
    
    bg, fg = vals[0], vals[1]
    
    comps_8 = find_all_8conn_components(grid, fg)
    if len(comps_8) <= 1:
        return grid
    
    comps_8.sort(key=len, reverse=True)
    main = comps_8[0]
    
    total_fg = sum(1 for r in range(H) for c in range(W) if grid[r][c] == fg)
    secondary_size = len(comps_8[1])
    
    if secondary_size / total_fg < 0.05:
        return grid
    
    enclosed = find_enclosed_4conn(grid, bg)
    
    if not enclosed:
        return grid
    
    to_fill = set()
    for r, c in enclosed:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in main:
                to_fill.add((r, c))
                break
    
    changed = True
    max_iter = 100
    iteration = 0
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        for r in range(H):
            for c in range(W):
                if grid[r][c] != bg or (r, c) in to_fill:
                    continue
                
                adj_fill_8 = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        if (r + dr, c + dc) in to_fill:
                            adj_fill_8 = True
                            break
                    if adj_fill_8:
                        break
                
                if not adj_fill_8:
                    continue
                
                adj_4 = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in main or (nr, nc) in to_fill:
                        adj_4 = True
                        break
                
                if adj_4:
                    to_fill.add((r, c))
                    changed = True
    
    result = [row[:] for row in grid]
    for r, c in to_fill:
        result[r][c] = fg
    
    return result
