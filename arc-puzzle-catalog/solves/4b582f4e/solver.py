def transform(grid):
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find connected components of non-bg pixels
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r, c) not in visited:
                queue = [(r, c)]
                visited.add((r, c))
                comp = [(r, c)]
                idx = 0
                while idx < len(queue):
                    cr, cc = queue[idx]
                    idx += 1
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] != bg:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                            comp.append((nr, nc))
                components.append(comp)

    for comp in components:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        h = max_r - min_r + 1
        w = max_c - min_c + 1

        colors = Counter(grid[r][c] for r, c in comp)
        majority_color = colors.most_common(1)[0][0]

        defect_pos = None
        defect_val = None

        if len(comp) == h * w:
            # Full rectangle - look for single pixel of different color
            for color, count in colors.items():
                if count == 1 and color != majority_color:
                    for r, c in comp:
                        if grid[r][c] == color:
                            defect_pos = (r, c)
                            defect_val = color
        elif len(comp) == h * w - 1:
            # Rectangle with 1 hole (bg pixel)
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if grid[r][c] == bg:
                        defect_pos = (r, c)
                        defect_val = bg

        if defect_pos is None:
            continue

        dr, dc = defect_pos

        on_top = (dr == min_r)
        on_bottom = (dr == max_r)
        on_left = (dc == min_c)
        on_right = (dc == max_c)

        if on_top:
            d = max_r - dr
            bc_start = max(0, dc - d)
            bc_end = min(cols - 1, dc + d)
            cross = [grid[dr][c] for c in range(bc_start, bc_end + 1)]
            for r in range(dr - 1, -1, -1):
                for j, c in enumerate(range(bc_start, bc_end + 1)):
                    if out[r][c] == bg:
                        out[r][c] = cross[j]
        elif on_bottom:
            d = dr - min_r
            bc_start = max(0, dc - d)
            bc_end = min(cols - 1, dc + d)
            cross = [grid[dr][c] for c in range(bc_start, bc_end + 1)]
            for r in range(dr + 1, rows):
                for j, c in enumerate(range(bc_start, bc_end + 1)):
                    if out[r][c] == bg:
                        out[r][c] = cross[j]
        elif on_left:
            d = max_c - dc
            br_start = max(0, dr - d)
            br_end = min(rows - 1, dr + d)
            cross = [grid[r][dc] for r in range(br_start, br_end + 1)]
            for c in range(dc - 1, -1, -1):
                for j, r in enumerate(range(br_start, br_end + 1)):
                    if out[r][c] == bg:
                        out[r][c] = cross[j]
        elif on_right:
            d = dc - min_c
            br_start = max(0, dr - d)
            br_end = min(rows - 1, dr + d)
            cross = [grid[r][dc] for r in range(br_start, br_end + 1)]
            for c in range(dc + 1, cols):
                for j, r in enumerate(range(br_start, br_end + 1)):
                    if out[r][c] == bg:
                        out[r][c] = cross[j]

    return out
