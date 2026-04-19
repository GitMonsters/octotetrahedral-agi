def transform(input_grid):
    from collections import Counter

    grid = [row[:] for row in input_grid]
    H = len(grid)
    W = len(grid[0])

    # Background = most common color
    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]

    def find_segments(line):
        segs = []
        i = 0
        while i < len(line):
            if line[i] != bg:
                color = line[i]
                start = i
                while i < len(line) and line[i] == color:
                    i += 1
                segs.append((color, start, i - 1))
            else:
                i += 1
        return segs

    # Find bars: contiguous non-bg segments (len>=2) on grid edges
    bars = []
    for edge_row in [0, H - 1]:
        for color, s, e in find_segments(grid[edge_row]):
            if e - s + 1 >= 2:
                bars.append({'o': 'h', 'pos': edge_row, 's': s, 'e': e, 'color': color})
    for edge_col in [0, W - 1]:
        col_vals = [grid[r][edge_col] for r in range(H)]
        for color, s, e in find_segments(col_vals):
            if e - s + 1 >= 2:
                bars.append({'o': 'v', 'pos': edge_col, 's': s, 'e': e, 'color': color})

    bar_color = bars[0]['color']

    # Find dots (non-bg, non-bar-color)
    dots = [(r, c) for r in range(H) for c in range(W)
            if grid[r][c] != bg and grid[r][c] != bar_color]
    dot_color = grid[dots[0][0]][dots[0][1]]

    if bars[0]['o'] == 'h':
        # Horizontal bars on top/bottom edges
        ranges = [set(range(b['s'], b['e'] + 1)) for b in bars]
        dot_cols = set(c for _, c in dots)
        if len(dot_cols & ranges[1]) > len(dot_cols & ranges[0]):
            barB, barA = bars[1], bars[0]
        else:
            barB, barA = bars[0], bars[1]

        dot_map = {}
        for r, c in dots:
            if barB['s'] <= c <= barB['e']:
                dot_map[c] = r

        d = -1 if barB['pos'] > barA['pos'] else 1  # barB toward barA

        # Fill from barB toward each dot
        for col, dr in dot_map.items():
            r = barB['pos'] + d
            while r != dr:
                grid[r][col] = dot_color
                r += d
            grid[dr][col] = bg

        # Fill barA region: mapped columns, full extent away from barA
        fd = -d
        for col in dot_map:
            mc = barA['s'] + (col - barB['s'])
            if fd == 1:
                for r in range(barA['pos'] + 1, H):
                    grid[r][mc] = dot_color
            else:
                for r in range(barA['pos'] - 1, -1, -1):
                    grid[r][mc] = dot_color
    else:
        # Vertical bars on left/right edges
        ranges = [set(range(b['s'], b['e'] + 1)) for b in bars]
        dot_rows = set(r for r, _ in dots)
        if len(dot_rows & ranges[1]) > len(dot_rows & ranges[0]):
            barB, barA = bars[1], bars[0]
        else:
            barB, barA = bars[0], bars[1]

        dot_map = {}
        for r, c in dots:
            if barB['s'] <= r <= barB['e']:
                dot_map[r] = c

        d = -1 if barB['pos'] > barA['pos'] else 1  # barB toward barA

        # Fill from barB toward each dot
        for row, dc in dot_map.items():
            c = barB['pos'] + d
            while c != dc:
                grid[row][c] = dot_color
                c += d
            grid[row][dc] = bg

        # Fill barA region: mapped rows, full extent away from barA
        fd = -d
        for row in dot_map:
            mr = barA['s'] + (row - barB['s'])
            if fd == 1:
                for c in range(barA['pos'] + 1, W):
                    grid[mr][c] = dot_color
            else:
                for c in range(barA['pos'] - 1, -1, -1):
                    grid[mr][c] = dot_color

    return grid
