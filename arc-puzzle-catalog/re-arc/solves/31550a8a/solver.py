from collections import deque


def find_blobs(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = grid[0][0]
    visited = [[False] * cols for _ in range(rows)]
    blobs = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and grid[nr][nc] != bg
                                and not visited[nr][nc]):
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                min_r = min(p[0] for p in cells)
                max_r = max(p[0] for p in cells)
                min_c = min(p[1] for p in cells)
                max_c = max(p[1] for p in cells)
                blobs.append({
                    "min_r": min_r, "max_r": max_r,
                    "min_c": min_c, "max_c": max_c,
                    "h": max_r - min_r + 1,
                    "w": max_c - min_c + 1,
                })
    seen = set()
    unique = []
    for b in blobs:
        key = (b["min_r"], b["max_r"], b["min_c"], b["max_c"])
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = grid[0][0]
    blobs = find_blobs(grid)
    result = [list(row) for row in grid]

    for b in blobs:
        r1, r2 = b["min_r"], b["max_r"]
        c1, c2 = b["min_c"], b["max_c"]
        # Number of concentric rings = max(h, w) // 2
        # (2x2 -> 1 ring, 4x4 -> 2 rings, 6x6 -> 3 rings)
        num_rings = max(b["h"], b["w"]) // 2

        # Draw border rings at distances 1, 2, ..., num_rings
        for k in range(1, num_rings + 1):
            box_r1 = max(0, r1 - k)
            box_r2 = min(rows - 1, r2 + k)
            box_c1 = max(0, c1 - k)
            box_c2 = min(cols - 1, c2 + k)
            for r in range(box_r1, box_r2 + 1):
                for c in range(box_c1, box_c2 + 1):
                    if (r == box_r1 or r == box_r2
                            or c == box_c1 or c == box_c2):
                        if result[r][c] == bg:
                            result[r][c] = 7

        # Downward tail from just below the outermost ring to grid bottom
        tail_start = r2 + num_rings + 1
        for r in range(tail_start, rows):
            for c in range(c1, c2 + 1):
                if result[r][c] == bg:
                    result[r][c] = 7

    return result
