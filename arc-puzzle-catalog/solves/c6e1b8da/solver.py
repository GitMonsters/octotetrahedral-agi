"""
Solver for ARC puzzle c6e1b8da.

Rule: The grid contains colored rectangular blocks. Some blocks have a 1-cell-wide
"connector" (protrusion) extending from one edge. In the output:
- The connector is removed.
- The rectangle shifts in the connector direction by the connector's length.
- Stationary rectangles (no connector) stay in place.
- Moving shapes render on top of stationary shapes.
"""
from collections import Counter
from typing import List


def solve(input_grid: List[List[int]]) -> List[List[int]]:
    H = len(input_grid)
    W = len(input_grid[0])

    colors = set()
    for row in input_grid:
        for cell in row:
            if cell != 0:
                colors.add(cell)

    # Classify each color as stationary or moving
    shape_info = {}  # color -> (type, rect, (dr, dc))

    for c in colors:
        cells = []
        for r in range(H):
            for cc in range(W):
                if input_grid[r][cc] == c:
                    cells.append((r, cc))
        if not cells:
            continue

        rs = [r for r, _ in cells]
        cs = [cc for _, cc in cells]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)

        # Perfect rectangle — stationary
        if len(cells) == bbox_area:
            shape_info[c] = ('stationary', (rmin, rmax, cmin, cmax), (0, 0))
            continue

        # Check if all gap cells in bbox are non-zero (hidden by other shapes)
        has_zero_gap = False
        for r in range(rmin, rmax + 1):
            for cc in range(cmin, cmax + 1):
                if input_grid[r][cc] != c and input_grid[r][cc] == 0:
                    has_zero_gap = True
                    break
            if has_zero_gap:
                break

        if not has_zero_gap:
            # Stationary, partially hidden by other shapes
            shape_info[c] = ('stationary', (rmin, rmax, cmin, cmax), (0, 0))
            continue

        # Has connector — identify direction and length
        row_dict: dict = {}
        for r, cc in cells:
            row_dict.setdefault(r, []).append(cc)

        row_spans = {}
        for r in sorted(row_dict):
            cs_list = sorted(row_dict[r])
            row_spans[r] = (min(cs_list), max(cs_list), len(cs_list))

        widths = [v[2] for v in row_spans.values()]
        wc = Counter(widths)
        main_w = wc.most_common(1)[0][0]

        main_rows = sorted([r for r in row_spans if row_spans[r][2] == main_w])
        extra_rows = sorted([r for r in row_spans if row_spans[r][2] != main_w])

        main_cmin = row_spans[main_rows[0]][0]
        main_cmax = row_spans[main_rows[0]][1]

        dr, dc = 0, 0
        rect = None

        if len(extra_rows) == 1:
            # Horizontal connector (one row wider than the rest)
            er = extra_rows[0]
            er_cmin, er_cmax, _ = row_spans[er]
            if er_cmin == main_cmin and er_cmax > main_cmax:
                dc = er_cmax - main_cmax   # connector goes right
            elif er_cmax == main_cmax and er_cmin < main_cmin:
                dc = -(main_cmin - er_cmin)  # connector goes left
            all_rows = main_rows + extra_rows
            rect = (min(all_rows), max(all_rows), main_cmin, main_cmax)

        elif extra_rows and all(row_spans[r][2] == 1 for r in extra_rows):
            # Vertical connector (extra rows with width 1)
            if max(extra_rows) > max(main_rows):
                dr = len(extra_rows)   # connector goes down
            else:
                dr = -len(extra_rows)  # connector goes up
            rect = (min(main_rows), max(main_rows), main_cmin, main_cmax)

        if rect and (dr != 0 or dc != 0):
            shape_info[c] = ('moving', rect, (dr, dc))
        else:
            shape_info[c] = ('stationary', (rmin, rmax, cmin, cmax), (0, 0))

    # Render output
    output = [[0] * W for _ in range(H)]

    # Draw stationary shapes first
    for c, (stype, rect, _) in shape_info.items():
        if stype == 'stationary':
            rmin, rmax, cmin, cmax = rect
            for r in range(rmin, rmax + 1):
                for cc in range(cmin, cmax + 1):
                    output[r][cc] = c

    # Draw moving shapes on top
    for c, (stype, rect, shift) in shape_info.items():
        if stype == 'moving':
            rmin, rmax, cmin, cmax = rect
            dr, dc = shift
            for r in range(rmin + dr, rmax + dr + 1):
                for cc in range(cmin + dc, cmax + dc + 1):
                    if 0 <= r < H and 0 <= cc < W:
                        output[r][cc] = c

    return output
