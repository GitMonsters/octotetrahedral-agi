from collections import deque


def find_enclosed_4conn(grid, bg):
    """Find cells enclosed by non-bg using 4-connectivity"""
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    queue = deque()
    
    # Start flood fill from edges
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
    """Fill interior of closed outlines with 7"""
    grid = [list(row) for row in grid]
    H, W = len(grid), len(grid[0])
    
    # Find colors
    colors = set(c for row in grid for c in row)
    
    # Identify background (most common)
    flat = [c for row in grid for c in row]
    bg = max(colors, key=flat.count)
    
    # Find enclosed cells
    enclosed = find_enclosed_4conn(grid, bg)
    
    # Fill with 7
    for r, c in enclosed:
        grid[r][c] = 7
    
    return grid
