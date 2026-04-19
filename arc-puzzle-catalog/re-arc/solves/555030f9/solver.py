def transform(grid):
    """
    Transformation rules:
    1. Framed rectangles (>=3x3, 1-cell border color A, interior color B):
       - Swap A and B within the rectangle
       - Extend outward with color A by interior dimensions (cross shape)
    2. Loose blobs (<3x3 in some dimension, single color C):
       - Add 1-cell border of C around blob
       - Original blob cells become background
    """
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter

    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find connected components of non-bg cells (8-connectivity)
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != bg:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                components.append(comp)

    out = [[bg] * cols for _ in range(rows)]

    for comp in components:
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)
        h = max_r - min_r + 1
        w = max_c - min_c + 1

        is_framed = False
        if h >= 3 and w >= 3:
            A = grid[min_r][min_c]  # border color
            is_framed = True

            # Verify border
            for c in range(min_c, max_c + 1):
                if grid[min_r][c] != A or grid[max_r][c] != A:
                    is_framed = False
                    break
            if is_framed:
                for r in range(min_r, max_r + 1):
                    if grid[r][min_c] != A or grid[r][max_c] != A:
                        is_framed = False
                        break
            if is_framed:
                B = grid[min_r + 1][min_c + 1]  # interior color (may be bg)
                for r in range(min_r + 1, max_r):
                    for c in range(min_c + 1, max_c):
                        if grid[r][c] != B:
                            is_framed = False
                            break
                    if not is_framed:
                        break

        if is_framed:
            ih = h - 2
            iw = w - 2

            # Swap A and B within rectangle
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if grid[r][c] == A:
                        out[r][c] = B
                    else:
                        out[r][c] = A

            # Extend with original border color A
            # Above
            for r in range(max(0, min_r - ih), min_r):
                for c in range(min_c, max_c + 1):
                    out[r][c] = A
            # Below
            for r in range(max_r + 1, min(rows, max_r + 1 + ih)):
                for c in range(min_c, max_c + 1):
                    out[r][c] = A
            # Left
            for c in range(max(0, min_c - iw), min_c):
                for r in range(min_r, max_r + 1):
                    out[r][c] = A
            # Right
            for c in range(max_c + 1, min(cols, max_c + 1 + iw)):
                for r in range(min_r, max_r + 1):
                    out[r][c] = A
        else:
            # Loose blob: frame it
            color = grid[comp[0][0]][comp[0][1]]
            nr1 = max(0, min_r - 1)
            nc1 = max(0, min_c - 1)
            nr2 = min(rows - 1, max_r + 1)
            nc2 = min(cols - 1, max_c + 1)

            for r in range(nr1, nr2 + 1):
                for c in range(nc1, nc2 + 1):
                    if r == nr1 or r == nr2 or c == nc1 or c == nc2:
                        out[r][c] = color
                    else:
                        out[r][c] = bg

    return out
