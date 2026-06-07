from collections import deque


def _blob_color(grid: list[list[int]]) -> int:
    """Return the color whose single largest connected component is the biggest."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    best_color, best_size = None, 0

    for sr in range(rows):
        for sc in range(cols):
            if visited[sr][sc]:
                continue
            color = grid[sr][sc]
            q = deque([(sr, sc)])
            visited[sr][sc] = True
            size = 0
            while q:
                r, c = q.popleft()
                size += 1
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                        visited[nr][nc] = True
                        q.append((nr, nc))
            if size > best_size:
                best_size = size
                best_color = color

    return best_color


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    blob_color = _blob_color(grid)

    # Copy grid then mirror every blob cell about the vertical centre axis
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == blob_color:
                mirror_c = cols - 1 - c
                if result[r][mirror_c] != blob_color:
                    result[r][mirror_c] = blob_color

    return result
