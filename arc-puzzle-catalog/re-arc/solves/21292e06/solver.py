from collections import Counter


def transform(grid):
    grid = [list(row) for row in grid]
    R = len(grid)
    C = len(grid[0])

    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    visited = [[False] * C for _ in range(R)]
    components = []

    for r in range(R):
        for c in range(C):
            if visited[r][c] or grid[r][c] == bg:
                continue
            cells = []
            stack = [(r, c)]
            while stack:
                nr, nc = stack.pop()
                if nr < 0 or nr >= R or nc < 0 or nc >= C:
                    continue
                if visited[nr][nc] or grid[nr][nc] == bg:
                    continue
                visited[nr][nc] = True
                cells.append((nr, nc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((nr + dr, nc + dc))
            if not cells:
                continue

            min_r = min(row for row, _ in cells)
            max_r = max(row for row, _ in cells)
            min_c = min(col for _, col in cells)
            max_c = max(col for _, col in cells)
            H = max_r - min_r + 1
            W = max_c - min_c + 1

            bbox_vals = [
                grid[rr][cc]
                for rr in range(min_r, max_r + 1)
                for cc in range(min_c, max_c + 1)
                if grid[rr][cc] != bg
            ]
            if not bbox_vals:
                continue
            fill = Counter(bbox_vals).most_common(1)[0][0]

            marker_pos = None
            marker_val = None
            for rr in range(min_r, max_r + 1):
                for cc in range(min_c, max_c + 1):
                    v = grid[rr][cc]
                    if v != fill:
                        marker_pos = (rr, cc)
                        marker_val = v
                        break
                if marker_pos:
                    break

            components.append({
                "fill": fill,
                "marker_pos": marker_pos,
                "marker_val": marker_val,
                "min_r": min_r, "max_r": max_r,
                "min_c": min_c, "max_c": max_c,
                "H": H, "W": W,
            })

    def paint_vertical(axis_c, fill, mv, row_start, row_end, half):
        for row in range(row_start, row_end):
            for dc in range(-half, half + 1):
                cc = axis_c + dc
                if 0 <= cc < C:
                    grid[row][cc] = mv if dc == 0 else fill

    def paint_horizontal(axis_r, fill, mv, col_start, col_end, half):
        for col in range(col_start, col_end):
            for dr in range(-half, half + 1):
                rr = axis_r + dr
                if 0 <= rr < R:
                    grid[rr][col] = mv if dr == 0 else fill

    for comp in components:
        if comp["marker_pos"] is None:
            continue
        mr, mc = comp["marker_pos"]
        fill, mv = comp["fill"], comp["marker_val"]
        min_r, max_r = comp["min_r"], comp["max_r"]
        min_c, max_c = comp["min_c"], comp["max_c"]
        H, W = comp["H"], comp["W"]
        half = min(H, W) - 1

        if mr == min_r:
            paint_vertical(mc, fill, mv, 0, min_r, half)
        elif mr == max_r:
            paint_vertical(mc, fill, mv, max_r + 1, R, half)
        elif mc == min_c:
            paint_horizontal(mr, fill, mv, 0, min_c, half)
        elif mc == max_c:
            paint_horizontal(mr, fill, mv, max_c + 1, C, half)

    for comp in components:
        if comp["marker_pos"] is not None:
            continue
        fill = comp["fill"]
        min_r, max_r = comp["min_r"], comp["max_r"]
        min_c, max_c = comp["min_c"], comp["max_c"]
        H, W = comp["H"], comp["W"]
        half = min(H, W) - 1

        trigger = None
        for other in components:
            if other is not comp and other["marker_val"] == fill:
                trigger = other
                break
        if trigger is None:
            continue

        tmr, tmc = trigger["marker_pos"]
        tmin_r, tmax_r = trigger["min_r"], trigger["max_r"]

        if tmr == tmin_r or tmr == tmax_r:
            axis_col = tmc
            if H == 1 and W == 1:
                r0, c0 = min_r, min_c
                if c0 >= axis_col:
                    for cc in range(c0 + 1, C):
                        grid[r0][cc] = fill
                else:
                    for cc in range(0, c0):
                        grid[r0][cc] = fill
            else:
                center_r = (min_r + max_r) // 2
                if max_c < axis_col:
                    for dr in range(-half, half + 1):
                        rr = center_r + dr
                        if 0 <= rr < R:
                            for cc in range(max_c + 1, C):
                                grid[rr][cc] = fill
                else:
                    for dr in range(-half, half + 1):
                        rr = center_r + dr
                        if 0 <= rr < R:
                            for cc in range(0, min_c):
                                grid[rr][cc] = fill
        else:
            axis_row = tmr
            if H == 1 and W == 1:
                r0, c0 = min_r, min_c
                if r0 >= axis_row:
                    for rr in range(r0 + 1, R):
                        grid[rr][c0] = fill
                else:
                    for rr in range(0, r0):
                        grid[rr][c0] = fill
            else:
                center_c = (min_c + max_c) // 2
                if max_r < axis_row:
                    for dc in range(-half, half + 1):
                        cc = center_c + dc
                        if 0 <= cc < C:
                            for rr in range(max_r + 1, R):
                                grid[rr][cc] = fill
                else:
                    for dc in range(-half, half + 1):
                        cc = center_c + dc
                        if 0 <= cc < C:
                            for rr in range(0, min_r):
                                grid[rr][cc] = fill

    return [list(row) for row in grid]
