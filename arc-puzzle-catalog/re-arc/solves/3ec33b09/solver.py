def transform(input_grid):
    from collections import Counter, deque

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[input_grid[r][c]] += 1
    bg = color_counts.most_common(1)[0][0]

    output = [[bg] * cols for _ in range(rows)]

    # Find connected components of non-bg cells (4-connected)
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if visited[r][c] or input_grid[r][c] == bg:
                continue
            comp = []
            queue = deque([(r, c)])
            visited[r][c] = True
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and input_grid[nr][nc] != bg:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
            components.append(comp)

    for comp in components:
        min_r = min(p[0] for p in comp)
        max_r = max(p[0] for p in comp)
        min_c = min(p[1] for p in comp)
        max_c = max(p[1] for p in comp)
        bh = max_r - min_r + 1
        bw = max_c - min_c + 1

        # Check if it's a rectangular frame (border B, fill F, at least 3x3)
        is_frame = False
        B = F = None
        if bh >= 3 and bw >= 3:
            B = input_grid[min_r][min_c]
            valid = True
            for cc in range(min_c, max_c + 1):
                if input_grid[min_r][cc] != B or input_grid[max_r][cc] != B:
                    valid = False
                    break
            if valid:
                for rr in range(min_r, max_r + 1):
                    if input_grid[rr][min_c] != B or input_grid[rr][max_c] != B:
                        valid = False
                        break
            if valid:
                F = input_grid[min_r + 1][min_c + 1]
                if F != B:
                    for rr in range(min_r + 1, max_r):
                        for cc in range(min_c + 1, max_c):
                            if input_grid[rr][cc] != F:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        is_frame = True
                else:
                    valid = False

        if is_frame:
            # Expand frame: swap B<->F inside, extend border outward
            h, w = bh, bw
            ih, iw = h - 2, w - 2  # interior dimensions
            new_r = min_r - ih
            new_c = min_c - iw
            new_h = h + 2 * ih
            new_w = w + 2 * iw

            for dr in range(new_h):
                for dc in range(new_w):
                    nr = new_r + dr
                    nc = new_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if dr < ih:  # top extension
                            if iw <= dc < iw + w:
                                output[nr][nc] = B
                        elif dr < ih + h:  # middle (original frame rows)
                            if dc < iw:
                                output[nr][nc] = B
                            elif dc < iw + w:
                                orig_val = input_grid[min_r + (dr - ih)][min_c + (dc - iw)]
                                output[nr][nc] = F if orig_val == B else B
                            else:
                                output[nr][nc] = B
                        else:  # bottom extension
                            if iw <= dc < iw + w:
                                output[nr][nc] = B
        else:
            # Solid block of single color -> wrap in a frame
            colors = set(input_grid[p[0]][p[1]] for p in comp)
            if len(colors) == 1:
                C = next(iter(colors))
                fr, fc = min_r - 1, min_c - 1
                fh, fw = bh + 2, bw + 2
                for dr in range(fh):
                    for dc in range(fw):
                        nr, nc = fr + dr, fc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if dr == 0 or dr == fh - 1 or dc == 0 or dc == fw - 1:
                                output[nr][nc] = C

    return output
