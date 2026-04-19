from collections import Counter


def find_components(grid, color, rows, cols):
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == color and not visited[r][c]:
                queue = [(r, c)]
                visited[r][c] = True
                cells = set()
                cells.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            cells.add((nr, nc))
                r1 = min(x[0] for x in cells)
                r2 = max(x[0] for x in cells)
                c1 = min(x[1] for x in cells)
                c2 = max(x[1] for x in cells)
                is_rect = len(cells) == (r2 - r1 + 1) * (c2 - c1 + 1)
                components.append({
                    "cells": cells,
                    "r1": r1, "c1": c1, "r2": r2, "c2": c2,
                    "h": r2 - r1 + 1, "w": c2 - c1 + 1,
                    "is_rect": is_rect,
                })
    return components


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    counter = Counter()
    for row in grid:
        counter.update(row)
    bg = counter.most_common(1)[0][0]
    non_bg = sorted(set(counter.keys()) - {bg})

    if len(non_bg) >= 2:
        frame_color = 0
        inner_color = [c for c in non_bg if c != 0][0]
    else:
        frame_color = non_bg[0]
        inner_color = non_bg[0]

    same_color = (frame_color == inner_color)

    if same_color:
        all_comps = find_components(grid, frame_color, rows, cols)
        frame_comps = all_comps
        inner_comps = all_comps
    else:
        frame_comps = find_components(grid, frame_color, rows, cols)
        inner_comps = find_components(grid, inner_color, rows, cols)

    frame = None
    for fc in frame_comps:
        if not fc["is_rect"]:
            frame = fc
            break
    if frame is None:
        return result

    if same_color:
        fb = frame
        body_rows = []
        arm_rows = []
        for r in range(fb["r1"], fb["r2"] + 1):
            row_cols = sorted([cc for (cr, cc) in fb["cells"] if cr == r])
            if (row_cols and row_cols[0] == fb["c1"] and row_cols[-1] == fb["c2"]
                    and len(row_cols) == fb["c2"] - fb["c1"] + 1):
                body_rows.append(r)
            elif row_cols:
                arm_rows.append(r)

        if not body_rows or not arm_rows:
            return result

        body_r1, body_r2 = min(body_rows), max(body_rows)
        body_h = body_r2 - body_r1 + 1

        arm_cols_set = {cc for r in arm_rows for (cr, cc) in fb["cells"] if cr == r}
        arm_c1, arm_c2 = min(arm_cols_set), max(arm_cols_set)
        arm_r1, arm_r2 = min(arm_rows), max(arm_rows)

        inner_h = arm_r2 - arm_r1 + 1
        inner_w = arm_c2 - arm_c1 + 1
        ml = arm_c1 - fb["c1"]
        mr = fb["c2"] - arm_c2

        truncated_side_v = "top" if arm_r1 > body_r2 else ("bottom" if arm_r2 < body_r1 else None)

        rect_same_size = [c for c in inner_comps if c["is_rect"] and c["h"] == inner_h and c["w"] == inner_w]
        tb_total = body_h - inner_h

        best = None
        for b in range(tb_total, -1, -1):
            t = tb_total - b
            if truncated_side_v == "top" and t >= b:
                continue
            if truncated_side_v == "bottom" and b >= t:
                continue

            t_corr = max(t, b)
            b_corr = max(t, b)

            for lr_corrected in [False, True]:
                ml_c = max(ml, mr) if lr_corrected else ml
                mr_c = max(ml, mr) if lr_corrected else mr

                fr1 = arm_r1 - t_corr
                fr2 = arm_r2 + b_corr
                fc1 = arm_c1 - ml_c
                fc2 = arm_c2 + mr_c

                if fr1 < 0 or fr2 >= rows or fc1 < 0 or fc2 >= cols:
                    continue

                overlap = any(
                    oi["r1"] <= fr2 and oi["r2"] >= fr1 and oi["c1"] <= fc2 and oi["c2"] >= fc1
                    for oi in rect_same_size
                )
                if not overlap:
                    if truncated_side_v == "top":
                        oi_r1 = body_r1 + t
                    else:
                        oi_r1 = body_r2 - t - inner_h + 1
                    best = {
                        "old_inner": (oi_r1, arm_c1, oi_r1 + inner_h - 1, arm_c2),
                        "new_frame": (fr1, fc1, fr2, fc2),
                    }
                    break
            if best:
                break

        if best is None:
            return result

        oir1, oic1, oir2, oic2 = best["old_inner"]
        nfr1, nfc1, nfr2, nfc2 = best["new_frame"]

        for (r, c) in frame["cells"]:
            result[r][c] = bg
        for r in range(oir1, oir2 + 1):
            for c in range(oic1, oic2 + 1):
                result[r][c] = inner_color
        for r in range(nfr1, nfr2 + 1):
            for c in range(nfc1, nfc2 + 1):
                result[r][c] = frame_color

    else:
        inner_in_frame = None
        other_inners = []
        for ic in inner_comps:
            if (ic["r1"] >= frame["r1"] and ic["r2"] <= frame["r2"]
                    and ic["c1"] >= frame["c1"] and ic["c2"] <= frame["c2"]):
                inner_in_frame = ic
            else:
                other_inners.append(ic)

        if inner_in_frame is None:
            return result

        mt = inner_in_frame["r1"] - frame["r1"]
        mb = frame["r2"] - inner_in_frame["r2"]
        ml_val = inner_in_frame["c1"] - frame["c1"]
        mr_val = frame["c2"] - inner_in_frame["c2"]
        inner_h = inner_in_frame["h"]
        inner_w = inner_in_frame["w"]

        mt_c, mb_c = max(mt, mb), max(mt, mb)
        ml_c, mr_c = max(ml_val, mr_val), max(ml_val, mr_val)

        same_size = sorted(
            [ic for ic in other_inners if ic["h"] == inner_h and ic["w"] == inner_w],
            key=lambda ic: abs(ic["r1"] - inner_in_frame["r1"]) + abs(ic["c1"] - inner_in_frame["c1"])
        )

        placed = False
        for corrections in ["full", "tb_only", "lr_only"]:
            if placed:
                break
            if corrections == "full":
                mu = (mt_c, mb_c, ml_c, mr_c)
            elif corrections == "tb_only":
                if mt == mb:
                    continue
                mu = (mt_c, mb_c, ml_val, mr_val)
            else:
                if ml_val == mr_val:
                    continue
                mu = (mt, mb, ml_c, mr_c)
            mt_u, mb_u, ml_u, mr_u = mu

            for ic in same_size:
                fr1 = ic["r1"] - mt_u
                fr2 = ic["r2"] + mb_u
                fc1 = ic["c1"] - ml_u
                fc2 = ic["c2"] + mr_u

                if fr1 < 0 or fr2 >= rows or fc1 < 0 or fc2 >= cols:
                    continue

                overlap = any(
                    oi["r1"] <= fr2 and oi["r2"] >= fr1 and oi["c1"] <= fc2 and oi["c2"] >= fc1
                    for oi in other_inners if oi is not ic
                )
                if overlap:
                    continue
                if (inner_in_frame["r1"] <= fr2 and inner_in_frame["r2"] >= fr1
                        and inner_in_frame["c1"] <= fc2 and inner_in_frame["c2"] >= fc1):
                    continue

                for (r, c) in frame["cells"]:
                    result[r][c] = bg
                for r in range(fr1, fr2 + 1):
                    for c in range(fc1, fc2 + 1):
                        if not (ic["r1"] <= r <= ic["r2"] and ic["c1"] <= c <= ic["c2"]):
                            result[r][c] = frame_color
                placed = True
                break

    return result
