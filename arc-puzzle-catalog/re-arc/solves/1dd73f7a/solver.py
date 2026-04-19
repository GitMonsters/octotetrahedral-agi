"""ARC task 1dd73f7a solver (pure Python, no imports).

Pattern:
- Find the largest axis-aligned rectangle whose four corners share a non-background
  color. Output is the crop of that rectangle.
- Outside that rectangle there is a small "template" pattern.
- Scale the template by integer factors so a unique template color aligns with an
  existing solid block of that color inside the crop, then overlay the scaled
  template onto the crop.

This reproduces all training examples exactly.
"""


def transform(grid):
    # ---------- basic helpers ----------
    def most_common_color(g):
        counts = {}
        for row in g:
            for v in row:
                counts[v] = counts.get(v, 0) + 1
        best_v = None
        best_n = -1
        for v, n in counts.items():
            if n > best_n:
                best_v = v
                best_n = n
        return best_v

    def crop(g, r1, r2, c1, c2):
        out = []
        for r in range(r1, r2 + 1):
            out.append(g[r][c1 : c2 + 1])
        return out

    # Find largest rectangle (by area) whose 4 corners are the same non-background color.
    def find_crop_rect(g, bg):
        h = len(g)
        w = len(g[0])
        best = None  # (area, r1, r2, c1, c2)

        for color in range(10):
            if color == bg:
                continue

            rowcols = {}
            pos = set()
            for r in range(h):
                row = g[r]
                for c in range(w):
                    if row[c] == color:
                        rowcols.setdefault(r, []).append(c)
                        pos.add((r, c))

            rows = sorted(rowcols)
            if len(rows) < 2:
                continue

            for i, r1 in enumerate(rows):
                cols1 = rowcols[r1]
                # unique + sorted columns
                seen = set()
                uniq_cols = []
                for c in cols1:
                    if c not in seen:
                        seen.add(c)
                        uniq_cols.append(c)
                uniq_cols.sort()
                if len(uniq_cols) < 2:
                    continue

                for a in range(len(uniq_cols)):
                    for b in range(a + 1, len(uniq_cols)):
                        c1 = uniq_cols[a]
                        c2 = uniq_cols[b]
                        for r2 in rows[i + 1 :]:
                            if (r2, c1) in pos and (r2, c2) in pos:
                                area = (r2 - r1 + 1) * (c2 - c1 + 1)
                                if best is None or area > best[0]:
                                    best = (area, r1, r2, c1, c2)

        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    def template_outside(g, bg, rect):
        r1, r2, c1, c2 = rect
        h = len(g)
        w = len(g[0])
        coords = []
        for r in range(h):
            in_r = r1 <= r <= r2
            row = g[r]
            for c in range(w):
                if not (in_r and c1 <= c <= c2) and row[c] != bg:
                    coords.append((r, c))

        if not coords:
            return None

        rmin = coords[0][0]
        rmax = coords[0][0]
        cmin = coords[0][1]
        cmax = coords[0][1]
        for r, c in coords[1:]:
            if r < rmin:
                rmin = r
            if r > rmax:
                rmax = r
            if c < cmin:
                cmin = c
            if c > cmax:
                cmax = c

        return crop(g, rmin, rmax, cmin, cmax)

    def find_unique_positions(T, bg):
        counts = {}
        pos = {}
        for i, row in enumerate(T):
            for j, v in enumerate(row):
                if v == bg:
                    continue
                counts[v] = counts.get(v, 0) + 1
                pos[v] = (i, j)
        uniques = []
        for v, cnt in counts.items():
            if cnt == 1:
                uniques.append(v)
        return uniques, pos

    # Find the largest 4-connected component of `color` that forms a solid rectangle.
    def largest_solid_component_bbox(g, color):
        H = len(g)
        W = len(g[0])
        visited = []
        for _ in range(H):
            visited.append([False] * W)

        best = None  # (size, rmin, rmax, cmin, cmax)

        for r in range(H):
            for c in range(W):
                if visited[r][c] or g[r][c] != color:
                    continue

                stack = [(r, c)]
                visited[r][c] = True

                cells = []
                rmin = rmax = r
                cmin = cmax = c

                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))

                    if rr < rmin:
                        rmin = rr
                    if rr > rmax:
                        rmax = rr
                    if cc < cmin:
                        cmin = cc
                    if cc > cmax:
                        cmax = cc

                    if rr > 0 and (not visited[rr - 1][cc]) and g[rr - 1][cc] == color:
                        visited[rr - 1][cc] = True
                        stack.append((rr - 1, cc))
                    if rr + 1 < H and (not visited[rr + 1][cc]) and g[rr + 1][cc] == color:
                        visited[rr + 1][cc] = True
                        stack.append((rr + 1, cc))
                    if cc > 0 and (not visited[rr][cc - 1]) and g[rr][cc - 1] == color:
                        visited[rr][cc - 1] = True
                        stack.append((rr, cc - 1))
                    if cc + 1 < W and (not visited[rr][cc + 1]) and g[rr][cc + 1] == color:
                        visited[rr][cc + 1] = True
                        stack.append((rr, cc + 1))

                area = (rmax - rmin + 1) * (cmax - cmin + 1)
                if area != len(cells):
                    continue

                # Verify rectangle is solid
                solid = True
                for rr in range(rmin, rmax + 1):
                    row = g[rr]
                    for cc in range(cmin, cmax + 1):
                        if row[cc] != color:
                            solid = False
                            break
                    if not solid:
                        break

                if not solid:
                    continue

                size = len(cells)
                if best is None or size > best[0]:
                    best = (size, rmin, rmax, cmin, cmax)

        return best

    def overlay_scaled(crop_grid, T, pr, pc, sy, sx):
        out = []
        for row in crop_grid:
            out.append(row[:])

        th = len(T)
        tw = len(T[0])
        for i in range(th):
            for j in range(tw):
                v = T[i][j]
                r0 = pr + i * sy
                c0 = pc + j * sx
                for rr in range(r0, r0 + sy):
                    row = out[rr]
                    for cc in range(c0, c0 + sx):
                        row[cc] = v

        return out

    def choose_overlay(crop_grid, T, bg):
        H = len(crop_grid)
        W = len(crop_grid[0])
        th = len(T)
        tw = len(T[0])

        uniques, pos = find_unique_positions(T, bg)
        best = None  # (score_tuple, overlaid_grid)

        # Prefer anchoring on a unique template color that exists as a solid block in the crop.
        for u in uniques:
            comp = largest_solid_component_bbox(crop_grid, u)
            if comp is None:
                continue

            _, rmin, rmax, cmin, cmax = comp
            sy = rmax - rmin + 1
            sx = cmax - cmin + 1
            ti, tj = pos[u]

            pr = rmin - ti * sy
            pc = cmin - tj * sx

            if pr < 0 or pc < 0:
                continue
            if pr + th * sy > H or pc + tw * sx > W:
                continue

            over = overlay_scaled(crop_grid, T, pr, pc, sy, sx)

            preserved = 0
            for r in range(H):
                row_in = crop_grid[r]
                row_out = over[r]
                for c in range(W):
                    if row_in[c] != bg and row_out[c] == row_in[c]:
                        preserved += 1

            cand = (preserved, sy * sx, -pr, -pc)
            if best is None or cand > best[0]:
                best = (cand, over)

        if best is not None:
            return best[1]

        # Fallback: brute-force search, biased toward larger scale factors.
        best_over = None
        best_cand = None
        for sy in range(1, H // th + 1):
            for sx in range(1, W // tw + 1):
                if sy == 1 and sx == 1:
                    continue
                sh = th * sy
                sw = tw * sx
                for pr in range(0, H - sh + 1):
                    for pc in range(0, W - sw + 1):
                        over = overlay_scaled(crop_grid, T, pr, pc, sy, sx)

                        preserved = 0
                        for r in range(H):
                            row_in = crop_grid[r]
                            row_out = over[r]
                            for c in range(W):
                                if row_in[c] != bg and row_out[c] == row_in[c]:
                                    preserved += 1

                        cand = (preserved, sy * sx, -pr, -pc)
                        if best_cand is None or cand > best_cand:
                            best_cand = cand
                            best_over = over

        return best_over

    # ---------- solve ----------
    bg = most_common_color(grid)
    rect = find_crop_rect(grid, bg)
    if rect is None:
        return [row[:] for row in grid]

    cropped = crop(grid, rect[0], rect[1], rect[2], rect[3])
    template = template_outside(grid, bg, rect)
    if template is None:
        return cropped

    over = choose_overlay(cropped, template, bg)
    return over if over is not None else cropped
