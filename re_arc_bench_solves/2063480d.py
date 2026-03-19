from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(v for r in grid for v in r).most_common(1)[0][0]
    out = [row[:] for row in grid]

    bar_rows = []
    bar_cols = []
    bar_color = None
    for r in range(H):
        if all(grid[r][c] == grid[r][0] for c in range(W)) and grid[r][0] != bg:
            bar_rows.append(r)
            bar_color = grid[r][0]
    for c in range(W):
        if all(grid[r][c] == grid[0][c] for r in range(H)) and grid[0][c] != bg:
            bar_cols.append(c)
            bar_color = grid[0][c] if bar_color is None else bar_color

    dots = []
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != bg and v != bar_color:
                dots.append((r, c, v))

    if bar_cols and not bar_rows:
        for dr, dc, dv in dots:
            left_bar = max((bc for bc in bar_cols if bc < dc), default=None)
            right_bar = min((bc for bc in bar_cols if bc > dc), default=None)
            for bc in [left_bar, right_bar]:
                if bc is None:
                    continue
                for r2 in range(max(0, dr-1), min(H, dr+2)):
                    for c2 in range(max(0, bc-1), min(W, bc+2)):
                        if r2 == dr and c2 == bc:
                            out[r2][c2] = dv
                        else:
                            out[r2][c2] = bar_color
                if bc < dc:
                    for c in range(bc+2, dc+1):
                        if out[dr][c] == bg:
                            out[dr][c] = dv
                else:
                    for c in range(dc, bc-1):
                        if out[dr][c] == bg:
                            out[dr][c] = dv

    elif bar_rows and not bar_cols:
        for dr, dc, dv in dots:
            above_bar = max((br for br in bar_rows if br < dr), default=None)
            below_bar = min((br for br in bar_rows if br > dr), default=None)
            for br in [above_bar, below_bar]:
                if br is None:
                    continue
                for r2 in range(max(0, br-1), min(H, br+2)):
                    for c2 in range(max(0, dc-1), min(W, dc+2)):
                        if r2 == br and c2 == dc:
                            out[r2][c2] = dv
                        else:
                            out[r2][c2] = bar_color
                if br < dr:
                    for r in range(br+2, dr+1):
                        if out[r][dc] == bg:
                            out[r][dc] = dv
                else:
                    for r in range(dr, br-1):
                        if out[r][dc] == bg:
                            out[r][dc] = dv

    return out
