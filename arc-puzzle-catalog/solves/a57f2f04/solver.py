def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]

    rectangles = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8 and not visited[r][c]:
                c_end = c
                while c_end + 1 < cols and grid[r][c_end + 1] != 8:
                    c_end += 1
                r_end = r
                while r_end + 1 < rows and grid[r_end + 1][c] != 8:
                    r_end += 1
                for rr in range(r, r_end + 1):
                    for cc in range(c, c_end + 1):
                        visited[rr][cc] = True
                rectangles.append((r, c, r_end, c_end))

    for r_top, c_left, r_bot, c_right in rectangles:
        # Find colored (non-0, non-8) cells
        colored = []
        for rr in range(r_top, r_bot + 1):
            for cc in range(c_left, c_right + 1):
                if grid[rr][cc] != 0 and grid[rr][cc] != 8:
                    colored.append((rr, cc))

        if not colored:
            continue

        cr_min = min(r for r, _ in colored)
        cr_max = max(r for r, _ in colored)
        cc_min = min(c for _, c in colored)
        cc_max = max(c for _, c in colored)

        # Determine corner and pattern tile bounds
        at_top = cr_min == r_top
        at_left = cc_min == c_left

        pr_top = r_top if at_top else cr_min
        pr_bot = cr_max if at_top else r_bot
        pc_left = c_left if at_left else cc_min
        pc_right = cc_max if at_left else c_right

        pH = pr_bot - pr_top + 1
        pW = pc_right - pc_left + 1

        # Extract pattern tile
        pattern = []
        for rr in range(pr_top, pr_bot + 1):
            pattern.append([grid[rr][cc] for cc in range(pc_left, pc_right + 1)])

        # Tile pattern across the rectangle
        for rr in range(r_top, r_bot + 1):
            for cc in range(c_left, c_right + 1):
                result[rr][cc] = pattern[(rr - r_top) % pH][(cc - c_left) % pW]

    return result
