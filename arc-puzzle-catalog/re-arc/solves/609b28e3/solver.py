from collections import Counter


def find_background(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]


def find_shapes(grid, bg):
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    shapes = []

    def bfs(sr, sc):
        comp = []
        queue = [(sr, sc)]
        visited[sr][sc] = True
        while queue:
            cr, cc = queue.pop(0)
            comp.append((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and not visited[nr][nc]
                        and grid[nr][nc] != bg):
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return comp

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                shapes.append(bfs(r, c))
    return shapes


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = find_background(grid)

    out = [[bg] * cols for _ in range(rows)]

    for comp in find_shapes(grid, bg):
        rs = [cell[0] for cell in comp]
        cs = [cell[1] for cell in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        H = max_r - min_r + 1
        W = max_c - min_c + 1
        ih = H - 2
        iw = W - 2

        B = grid[min_r][min_c]
        F = grid[min_r + 1][min_c + 1] if ih > 0 and iw > 0 else bg

        # Swapped shape: border positions -> F, interior positions -> B
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                is_border = (r == min_r or r == max_r
                             or c == min_c or c == max_c)
                out[r][c] = F if is_border else B

        # Frame of B: ih rows above/below (full width), iw cols left/right (full height)
        for dr in range(1, ih + 1):
            nr = min_r - dr
            if 0 <= nr < rows:
                for c in range(min_c, max_c + 1):
                    out[nr][c] = B
        for dr in range(1, ih + 1):
            nr = max_r + dr
            if 0 <= nr < rows:
                for c in range(min_c, max_c + 1):
                    out[nr][c] = B
        for dc in range(1, iw + 1):
            nc = min_c - dc
            if 0 <= nc < cols:
                for r in range(min_r, max_r + 1):
                    out[r][nc] = B
        for dc in range(1, iw + 1):
            nc = max_c + dc
            if 0 <= nc < cols:
                for r in range(min_r, max_r + 1):
                    out[r][nc] = B

    return out
