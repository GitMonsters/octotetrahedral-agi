from collections import Counter


def _copy_grid(grid):
    return [row[:] for row in grid]


def _paint(out, top, left, bottom, right, color):
    if top > bottom or left > right:
        return
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            out[r][c] = color


def _top_uniform_height(grid, color):
    h = 0
    width = len(grid[0])
    for row in grid:
        if all(v == color for v in row[:width]):
            h += 1
        else:
            break
    return h


def _full_height_run(grid, color):
    height = len(grid)
    width = len(grid[0])
    best = None
    start = None
    for c in range(width):
        full = all(grid[r][c] == color for r in range(height))
        if full:
            if start is None:
                start = c
        elif start is not None:
            cand = (start, c - 1)
            if best is None or cand[1] - cand[0] > best[1] - best[0]:
                best = cand
            start = None
    if start is not None:
        cand = (start, width - 1)
        if best is None or cand[1] - cand[0] > best[1] - best[0]:
            best = cand
    return best


def _consecutive_top_rows_all(grid, color, left, right):
    h = 0
    for row in grid:
        if all(v == color for v in row[left:right + 1]):
            h += 1
        else:
            break
    return h


def _prefix_for_color(grid, color):
    h, w = len(grid), len(grid[0])
    pref = [[0] * (w + 1) for _ in range(h + 1)]
    for r in range(h):
        row_sum = 0
        for c in range(w):
            row_sum += 1 if grid[r][c] == color else 0
            pref[r + 1][c + 1] = pref[r][c + 1] + row_sum
    return pref


def _rect_count(pref, top, left, bottom, right):
    return (
        pref[bottom + 1][right + 1]
        - pref[top][right + 1]
        - pref[bottom + 1][left]
        + pref[top][left]
    )


def _all_solid_rectangles(grid, color):
    h, w = len(grid), len(grid[0])
    pref = _prefix_for_color(grid, color)
    rects = []
    for top in range(h):
        for left in range(w):
            for bottom in range(top, h):
                for right in range(left, w):
                    area = (bottom - top + 1) * (right - left + 1)
                    if _rect_count(pref, top, left, bottom, right) == area:
                        rects.append((top, left, bottom, right))
    return rects


def _choose_bar_rectangles(grid):
    h, w = len(grid), len(grid[0])
    counts = Counter(v for row in grid for v in row if v != 4)
    colors = [color for color, _ in counts.most_common()]
    best = None

    for color in colors:
        rects = _all_solid_rectangles(grid, color)
        verticals = [r for r in rects if (r[3] - r[1] + 1) >= 3]
        if not verticals:
            continue

        vertical = max(
            verticals,
            key=lambda r: (
                r[2] - r[0] + 1,
                (r[3] - r[1] + 1) >= 5,
                (r[2] - r[0] + 1) * (r[3] - r[1] + 1),
            ),
        )

        preferred = []
        fallback = []
        for r in rects:
            height = r[2] - r[0] + 1
            width = r[3] - r[1] + 1
            if height < 3:
                continue
            target = preferred if height >= 4 else fallback
            target.append(r)
        horizontals = preferred or fallback
        if not horizontals:
            continue

        def hscore(r):
            overlap = not (r[2] < vertical[0] or r[0] > vertical[2] or r[3] < vertical[1] or r[1] > vertical[3])
            return (
                r[2] - r[0] + 1 >= 4,
                overlap,
                r[3] - r[1] + 1,
                (r[2] - r[0] + 1) * (r[3] - r[1] + 1),
            )

        horizontal = max(horizontals, key=hscore)

        score = (
            (vertical[2] - vertical[0] + 1) * 10
            + (horizontal[3] - horizontal[1] + 1) * 10
            + (vertical[3] - vertical[1] + 1)
            + (horizontal[2] - horizontal[0] + 1)
        )
        if best is None or score > best[0]:
            best = (score, color, vertical, horizontal)

    return None if best is None else best[1:]


def _trim_by_edges(rect, height, width):
    top, left, bottom, right = rect
    if top > 0:
        top += 1
    if bottom < height - 1:
        bottom -= 1
    if left > 0:
        left += 1
    if right < width - 1:
        right -= 1
    return top, left, bottom, right


def transform(grid):
    out = _copy_grid(grid)
    h, w = len(grid), len(grid[0])

    if any(4 in row for row in grid):
        return out

    top_color = grid[0][0]
    top_h = _top_uniform_height(grid, top_color)
    full_run = _full_height_run(grid, top_color)

    if top_h >= 3 and full_run is not None:
        run_left, run_right = full_run
        if run_right - run_left + 1 > 3:
            inner_left, inner_right = run_left + 1, run_right - 1
        else:
            inner_left, inner_right = run_left, run_right

        _paint(out, 0, inner_left, h - 1, inner_right, 4)
        _paint(out, 0, 0, top_h - 2, w - 1, 4)

        right_h = _consecutive_top_rows_all(grid, top_color, inner_left, w - 1)
        _paint(out, 0, inner_left, right_h - 2, w - 1, 4)
        return out

    bars = _choose_bar_rectangles(grid)
    if bars is None:
        return out

    _, vertical, horizontal = bars
    v_top, v_left, v_bottom, v_right = _trim_by_edges(vertical, h, w)
    h_top, h_left, h_bottom, h_right = _trim_by_edges(horizontal, h, w)

    _paint(out, v_top, v_left, v_bottom, v_right, 4)
    _paint(out, h_top, h_left, h_bottom, h_right, 4)
    return out
