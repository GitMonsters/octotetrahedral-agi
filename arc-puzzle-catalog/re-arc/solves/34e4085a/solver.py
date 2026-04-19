def transform(input_grid):
    from collections import Counter
    from itertools import combinations

    grid = [row[:] for row in input_grid]
    R = len(grid)
    C = len(grid[0])

    # 1. Find background (most common color)
    flat = [grid[r][c] for r in range(R) for c in range(C)]
    bg = Counter(flat).most_common(1)[0][0]

    # 2. Find non-bg pixels by color
    color_pixels = {}
    for r in range(R):
        for c in range(C):
            v = grid[r][c]
            if v != bg:
                color_pixels.setdefault(v, []).append((r, c))

    # 3. Find 4 same-color pixels forming the largest rectangle (corner markers)
    best_area = 0
    best_corners = None
    corner_color = None

    for color, pixels in color_pixels.items():
        pixel_set = set(pixels)
        rows = sorted(set(r for r, c in pixels))
        cols = sorted(set(c for r, c in pixels))
        for r_lo, r_hi in combinations(rows, 2):
            for c_lo, c_hi in combinations(cols, 2):
                if ((r_lo, c_lo) in pixel_set and (r_lo, c_hi) in pixel_set and
                    (r_hi, c_lo) in pixel_set and (r_hi, c_hi) in pixel_set):
                    area = (r_hi - r_lo) * (c_hi - c_lo)
                    if area > best_area:
                        best_area = area
                        best_corners = (r_lo, c_lo, r_hi, c_hi)
                        corner_color = color

    r1, c1, r2, c2 = best_corners
    corner_pos = {(r1, c1), (r1, c2), (r2, c1), (r2, c2)}

    # 4. Extract rectangle
    rect_h = r2 - r1 + 1
    rect_w = c2 - c1 + 1
    rect = [[grid[r1 + dr][c1 + dc] for dc in range(rect_w)] for dr in range(rect_h)]

    # 5. Inside pixels (relative to rect, excluding corner positions) and key pixels (outside rect)
    inside = {}
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if (r, c) in corner_pos:
                continue
            v = grid[r][c]
            if v != bg:
                inside.setdefault(v, []).append((r - r1, c - c1))

    key_abs = {}
    for r in range(R):
        for c in range(C):
            if r1 <= r <= r2 and c1 <= c <= c2:
                continue
            v = grid[r][c]
            if v != bg:
                key_abs.setdefault(v, []).append((r, c))

    if not key_abs:
        return rect

    # Normalize key to its bounding box origin
    all_key_pts = [p for pts in key_abs.values() for p in pts]
    km_r = min(r for r, c in all_key_pts)
    km_c = min(c for r, c in all_key_pts)

    key = {}
    for color, pts in key_abs.items():
        key[color] = [(r - km_r, c - km_c) for r, c in pts]

    # 6-7. Determine scale and origin
    scale_r = scale_c = origin_r = origin_c = None

    # Strategy 1: color with exactly 1 key pixel and inside blocks
    for color in key:
        if color in inside and len(key[color]) == 1:
            kr, kc = key[color][0]
            ipx = inside[color]
            ir_min = min(r for r, c in ipx)
            ir_max = max(r for r, c in ipx)
            ic_min = min(c for r, c in ipx)
            ic_max = max(c for r, c in ipx)
            scale_r = ir_max - ir_min + 1
            scale_c = ic_max - ic_min + 1
            origin_r = ir_min - kr * scale_r
            origin_c = ic_min - kc * scale_c
            break

    # Strategy 2: color with multiple key pixels, use bbox ratio
    if scale_r is None:
        for color in key:
            if color not in inside:
                continue
            kpx = key[color]
            ipx = inside[color]
            kr_min = min(r for r, c in kpx)
            kr_max = max(r for r, c in kpx)
            kc_min = min(c for r, c in kpx)
            kc_max = max(c for r, c in kpx)
            kh = kr_max - kr_min + 1
            kw = kc_max - kc_min + 1

            ir_min = min(r for r, c in ipx)
            ir_max = max(r for r, c in ipx)
            ic_min = min(c for r, c in ipx)
            ic_max = max(c for r, c in ipx)
            ih = ir_max - ir_min + 1
            iw = ic_max - ic_min + 1

            if ih % kh == 0 and iw % kw == 0:
                scale_r = ih // kh
                scale_c = iw // kw
                origin_r = ir_min - kr_min * scale_r
                origin_c = ic_min - kc_min * scale_c
                break

    # Strategy 3: brute force match blocks to key pixels
    if scale_r is None:
        visited = set()
        blocks = []
        for color in inside:
            pset = set(inside[color])
            for r, c in inside[color]:
                if (color, r, c) in visited:
                    continue
                queue = [(r, c)]
                block = []
                while queue:
                    cr, cc = queue.pop(0)
                    if (color, cr, cc) in visited or (cr, cc) not in pset:
                        continue
                    visited.add((color, cr, cc))
                    block.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (color, nr, nc) not in visited and (nr, nc) in pset:
                            queue.append((nr, nc))
                if block:
                    br_min = min(r for r, c in block)
                    bc_min = min(c for r, c in block)
                    bh = max(r for r, c in block) - br_min + 1
                    bw = max(c for r, c in block) - bc_min + 1
                    blocks.append((color, br_min, bc_min, bh, bw))

        for bcolor, br, bc, bh, bw in blocks:
            if bcolor not in key:
                continue
            for kr, kc in key[bcolor]:
                sr, sc = bh, bw
                o_r = br - kr * sr
                o_c = bc - kc * sc
                valid = True
                for b2color, b2r, b2c, b2h, b2w in blocks:
                    if b2color not in key:
                        valid = False
                        break
                    matched = any(
                        o_r + k2r * sr == b2r and o_c + k2c * sc == b2c
                        for k2r, k2c in key[b2color]
                    )
                    if not matched:
                        valid = False
                        break
                if valid:
                    scale_r, scale_c = sr, sc
                    origin_r, origin_c = o_r, o_c
                    break
            if scale_r is not None:
                break

    if scale_r is None:
        return rect

    # 8. Fill all key positions into the output
    output = [row[:] for row in rect]
    for color, kpxs in key.items():
        for kr, kc in kpxs:
            sr = origin_r + kr * scale_r
            sc = origin_c + kc * scale_c
            for dr in range(scale_r):
                for dc in range(scale_c):
                    rr = sr + dr
                    cc = sc + dc
                    if 0 <= rr < rect_h and 0 <= cc < rect_w:
                        output[rr][cc] = color

    return output
