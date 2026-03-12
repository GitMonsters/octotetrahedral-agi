def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find the original rectangular block
    min_r = min_c = float('inf')
    max_r = max_c = -1
    color = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    h = max_r - min_r + 1
    w = max_c - min_c + 1
    other_color = 6 if color == 3 else 3

    out = [[0] * cols for _ in range(rows)]

    def place(r, c, bh, bw, col):
        for dr in range(bh):
            for dc in range(bw):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    out[rr][cc] = col

    # Place original block
    place(min_r, min_c, h, w, color)

    # Extend four diagonal chains outward, alternating transposed shape and color
    for dr_sign, dc_sign in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        cr, cc, ch, cw = min_r, min_c, h, w
        cur_col = other_color

        while True:
            nr = (cr - cw) if dr_sign < 0 else (cr + ch)
            nc = (cc - ch) if dc_sign < 0 else (cc + cw)
            nh, nw = cw, ch

            if nr >= rows or nr + nh <= 0:
                break
            if nc >= cols or nc + nw <= 0:
                break

            place(nr, nc, nh, nw, cur_col)

            cur_col = color if cur_col == other_color else other_color
            cr, cc, ch, cw = nr, nc, nh, nw

    return out
