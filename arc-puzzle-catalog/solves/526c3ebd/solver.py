"""
Solver for ARC task 526c3ebd.

The grid has content blocks separated by a large L/T/cross-shaped region of
pure background. The transformation fills the "interior" of this region with
color 9, shrinking by 1 cell from each side that borders content (not grid edges).

Algorithm:
1. Find background color (most frequent).
2. Find the largest all-bg rectangle (one arm of the L/T/cross).
3. Find the second arm by extending the first arm in orthogonal direction.
4. Compute interior of each arm (shrink 1 from content borders).
5. Fill union of interiors with 9.
"""
from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    output = [row[:] for row in grid]

    arm1 = _largest_bg_rect(grid, bg, rows, cols)
    if arm1 is None:
        return output

    arm2 = _find_second_arm(grid, bg, arm1, rows, cols)

    _fill_interior(output, arm1, rows, cols)
    if arm2:
        _fill_interior(output, arm2, rows, cols)

    return output


def _largest_bg_rect(grid, bg, rows, cols):
    """Find the largest all-bg rectangle using the histogram method."""
    height = [0] * cols
    best = None
    best_area = 0

    for r in range(rows):
        for c in range(cols):
            height[c] = height[c] + 1 if grid[r][c] == bg else 0

        stack = []
        for c in range(cols + 1):
            h = height[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                sc, sh = stack.pop()
                w = c - sc
                area = w * sh
                if area > best_area:
                    best_area = area
                    best = (r - sh + 1, sc, r, c - 1)
                start = sc
            stack.append((start, h))

    return best


def _find_second_arm(grid, bg, arm1, rows, cols):
    """Find a second all-bg rectangle that overlaps with arm1 and maximizes
    the union area, by extending arm1's column or row range."""
    ext_col = _extend_cols(grid, bg, arm1, rows, cols)
    ext_row = _extend_rows(grid, bg, arm1, rows, cols)

    a1_area = _rect_area(arm1)
    u_col = _union_area(arm1, ext_col) if ext_col else a1_area
    u_row = _union_area(arm1, ext_row) if ext_row else a1_area

    if u_col > a1_area and u_col >= u_row:
        return ext_col
    elif u_row > a1_area:
        return ext_row
    return None


def _extend_cols(grid, bg, arm1, rows, cols):
    """For each row in arm1, find the full bg column range including arm1's cols.
    Then find the best rectangle that extends arm1 horizontally."""
    a_r1, a_c1, a_r2, a_c2 = arm1

    left = {}
    right = {}
    for r in range(a_r1, a_r2 + 1):
        l = a_c1
        while l > 0 and grid[r][l - 1] == bg:
            l -= 1
        left[r] = l

        ri = a_c2
        while ri < cols - 1 and grid[r][ri + 1] == bg:
            ri += 1
        right[r] = ri

    l_vals = sorted(set(left.values()))
    r_vals = sorted(set(right.values()), reverse=True)

    best = None
    best_union = _rect_area(arm1)

    for l in l_vals:
        for ri in r_vals:
            if l >= a_c1 and ri <= a_c2:
                continue  # Not extending beyond arm1
            run_start = None
            run_end = None
            for r in range(a_r1, a_r2 + 1):
                if left[r] <= l and right[r] >= ri:
                    if run_start is None:
                        run_start = r
                    run_end = r
                else:
                    if run_start is not None:
                        cand = (run_start, l, run_end, ri)
                        u = _union_area(arm1, cand)
                        if u > best_union:
                            best_union = u
                            best = cand
                    run_start = None
            if run_start is not None:
                cand = (run_start, l, run_end, ri)
                u = _union_area(arm1, cand)
                if u > best_union:
                    best_union = u
                    best = cand

    return best


def _extend_rows(grid, bg, arm1, rows, cols):
    """For each col in arm1, find the full bg row range including arm1's rows.
    Then find the best rectangle that extends arm1 vertically."""
    a_r1, a_c1, a_r2, a_c2 = arm1

    up = {}
    down = {}
    for c in range(a_c1, a_c2 + 1):
        u = a_r1
        while u > 0 and grid[u - 1][c] == bg:
            u -= 1
        up[c] = u

        d = a_r2
        while d < rows - 1 and grid[d + 1][c] == bg:
            d += 1
        down[c] = d

    u_vals = sorted(set(up.values()))
    d_vals = sorted(set(down.values()), reverse=True)

    best = None
    best_union = _rect_area(arm1)

    for u in u_vals:
        for d in d_vals:
            if u >= a_r1 and d <= a_r2:
                continue  # Not extending beyond arm1
            run_start = None
            run_end = None
            for c in range(a_c1, a_c2 + 1):
                if up[c] <= u and down[c] >= d:
                    if run_start is None:
                        run_start = c
                    run_end = c
                else:
                    if run_start is not None:
                        cand = (u, run_start, d, run_end)
                        union = _union_area(arm1, cand)
                        if union > best_union:
                            best_union = union
                            best = cand
                    run_start = None
            if run_start is not None:
                cand = (u, run_start, d, run_end)
                union = _union_area(arm1, cand)
                if union > best_union:
                    best_union = union
                    best = cand

    return best


def _rect_area(rect):
    r1, c1, r2, c2 = rect
    return (r2 - r1 + 1) * (c2 - c1 + 1)


def _union_area(rect1, rect2):
    r1a, c1a, r2a, c2a = rect1
    r1b, c1b, r2b, c2b = rect2
    area_a = (r2a - r1a + 1) * (c2a - c1a + 1)
    area_b = (r2b - r1b + 1) * (c2b - c1b + 1)
    ir1 = max(r1a, r1b)
    ic1 = max(c1a, c1b)
    ir2 = min(r2a, r2b)
    ic2 = min(c2a, c2b)
    area_i = max(0, ir2 - ir1 + 1) * max(0, ic2 - ic1 + 1) if ir1 <= ir2 and ic1 <= ic2 else 0
    return area_a + area_b - area_i


def _fill_interior(output, arm, rows, cols):
    r1, c1, r2, c2 = arm
    ri1 = r1 + (1 if r1 > 0 else 0)
    ri2 = r2 - (1 if r2 < rows - 1 else 0)
    ci1 = c1 + (1 if c1 > 0 else 0)
    ci2 = c2 - (1 if c2 < cols - 1 else 0)
    for r in range(ri1, ri2 + 1):
        for c in range(ci1, ci2 + 1):
            output[r][c] = 9
