from collections import Counter


def _transpose(grid):
    return [list(row) for row in zip(*grid)]


def _background(grid):
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _full_rows(grid, bg):
    return [(r, row[0]) for r, row in enumerate(grid) if len(set(row)) == 1 and row[0] != bg]


def _place_distinct_rows(grid, guides, bg):
    h = len(grid)
    w = len(grid[0])
    out = [[bg] * w for _ in range(h)]
    guide_rows = {r for r, _ in guides}
    for r, color in guides:
        out[r] = [color] * w

    ordered = sorted(guides)
    for idx, (gr, color) in enumerate(ordered):
        outer_side = -1 if idx == 0 else 1
        inner_bound = ordered[idx + 1][0] if idx + 1 < len(ordered) else None
        lower_bound = ordered[idx - 1][0] if idx > 0 else None
        for c in range(w):
            rows = [r for r in range(h) if r != gr and r not in guide_rows and grid[r][c] == color]
            if not rows:
                continue
            src = min(rows, key=lambda r: (abs(r - gr), r))
            side = -1 if src < gr else 1
            dist = abs(src - gr)
            between_guides = False
            if side == 1 and inner_bound is not None and gr < src < inner_bound:
                between_guides = True
            if side == -1 and lower_bound is not None and lower_bound < src < gr:
                between_guides = True
            if idx == 0 and between_guides and dist > 4:
                continue
            if idx == 0 and side == outer_side and dist == 1:
                continue
            target = gr + side
            if 0 <= target < h:
                out[target][c] = color
    return out


def _solve_same_color_rows(grid, guides, bg):
    h = len(grid)
    w = len(grid[0])
    (g1, color), (g2, _) = sorted(guides)
    out = [[bg] * w for _ in range(h)]
    out[g1] = [color] * w
    out[g2] = [color] * w

    row_above = g1 - 1
    row_below_top = g1 + 1
    row_above_bottom = g2 - 1
    row_below = g2 + 1

    for c in range(w):
        above = [grid[r][c] for r in range(0, g1) if grid[r][c] != bg]
        middle = [grid[r][c] for r in range(g1 + 1, g2) if grid[r][c] != bg]
        below = [grid[r][c] for r in range(g2 + 1, h) if grid[r][c] != bg]

        above_guide = sum(v == color for v in above)
        middle_guide = sum(v == color for v in middle)
        below_guide = sum(v == color for v in below)
        above_has6 = 6 in above
        below_has6 = 6 in below
        below_has4 = 4 in below
        above_boundary = 0 <= row_above < h and grid[row_above][c] == color

        if 0 <= row_above < h and above_guide and (above_has6 or above_boundary):
            out[row_above][c] = color

        trigger_below_top = False
        if middle_guide:
            trigger_below_top = True
        elif below_guide and (below_guide > 1 or (below_has6 and not below_has4)):
            trigger_below_top = True
        if 0 <= row_below_top < h and trigger_below_top:
            out[row_below_top][c] = color

        if 0 <= row_above_bottom < h and middle and not trigger_below_top:
            out[row_above_bottom][c] = color

        if 0 <= row_below < h and below_guide and not (below_guide == 1 and below_has6 and not below_has4):
            out[row_below][c] = color

    return out


def transform(grid):
    bg = _background(grid)
    row_guides = _full_rows(grid, bg)
    if row_guides:
        if len(row_guides) == 2 and row_guides[0][1] == row_guides[1][1]:
            return _solve_same_color_rows(grid, row_guides, bg)
        return _place_distinct_rows(grid, sorted(row_guides), bg)

    tg = _transpose(grid)
    tbg = _background(tg)
    col_guides = _full_rows(tg, tbg)
    if len(col_guides) == 2 and col_guides[0][1] == col_guides[1][1]:
        return _transpose(_solve_same_color_rows(tg, col_guides, tbg))
    return _transpose(_place_distinct_rows(tg, sorted(col_guides), tbg))
