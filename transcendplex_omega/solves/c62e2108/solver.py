def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Rectangles tile/repeat toward 1-markers using modular pattern repetition.
    Horizontal markers (same cols as rect) → tile up/down.
    Vertical markers (same rows as rect) → tile left/right.
    """
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    # Find rectangle objects (connected components of color > 1)
    visited = [[False] * cols for _ in range(rows)]
    rectangles = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] > 1 and not visited[r][c]:
                color = grid[r][c]
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))

                min_r = min(p[0] for p in component)
                max_r = max(p[0] for p in component)
                min_c = min(p[1] for p in component)
                max_c = max(p[1] for p in component)

                h = max_r - min_r + 1
                w = max_c - min_c + 1
                pattern = []
                for pr in range(h):
                    row = []
                    for pc in range(w):
                        val = grid[min_r + pr][min_c + pc]
                        row.append(color if val == color else 0)
                    pattern.append(row)

                rectangles.append((min_r, min_c, max_r, max_c, color, pattern))

    # Find 1-marker groups (connected components of 1s)
    marker_cells = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1}
    marker_groups = []
    visited_markers: set[tuple[int, int]] = set()

    for cell in sorted(marker_cells):
        if cell in visited_markers:
            continue
        group = []
        stack = [cell]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited_markers or (cr, cc) not in marker_cells:
                continue
            visited_markers.add((cr, cc))
            group.append((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((cr + dr, cc + dc))

        rs = [p[0] for p in group]
        cs = [p[1] for p in group]
        if min(rs) == max(rs):
            marker_groups.append(('h', min(rs), min(cs), max(cs)))
        elif min(cs) == max(cs):
            marker_groups.append(('v', min(cs), min(rs), max(rs)))

    # Match markers to rectangles and tile
    for marker in marker_groups:
        if marker[0] == 'h':
            _, rm, mc1, mc2 = marker
            for r1, c1, r2, c2, color, pattern in rectangles:
                if c1 == mc1 and c2 == mc2:
                    h = r2 - r1 + 1
                    w = c2 - c1 + 1
                    if rm < r1:  # tile upward
                        for tr in range(rm, r1):
                            for tc in range(c1, c2 + 1):
                                output[tr][tc] = pattern[(tr - r1) % h][(tc - c1) % w]
                    elif rm > r2:  # tile downward
                        for tr in range(r2 + 1, rm + 1):
                            for tc in range(c1, c2 + 1):
                                output[tr][tc] = pattern[(tr - r1) % h][(tc - c1) % w]
        elif marker[0] == 'v':
            _, cm, mr1, mr2 = marker
            for r1, c1, r2, c2, color, pattern in rectangles:
                if r1 == mr1 and r2 == mr2:
                    h = r2 - r1 + 1
                    w = c2 - c1 + 1
                    if cm < c1:  # tile leftward
                        for tr in range(r1, r2 + 1):
                            for tc in range(cm, c1):
                                output[tr][tc] = pattern[(tr - r1) % h][(tc - c1) % w]
                    elif cm > c2:  # tile rightward
                        for tr in range(r1, r2 + 1):
                            for tc in range(c2 + 1, cm + 1):
                                output[tr][tc] = pattern[(tr - r1) % h][(tc - c1) % w]

    # Remove any remaining 1s
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 1:
                output[r][c] = 0

    return output
