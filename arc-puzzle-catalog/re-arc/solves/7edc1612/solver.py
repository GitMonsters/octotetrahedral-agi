from collections import Counter


def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color
    flat = [v for row in input_grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find non-background cells
    cells = {}
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                cells[(r, c)] = input_grid[r][c]

    if not cells:
        return [row[:] for row in input_grid]

    # C4 rotation transforms about center (cr, cc)
    def c4(r, c, cr, cc):
        dr, dc = r - cr, c - cc
        return [
            (cr + dr, cc + dc),
            (cr + dc, cc - dr),
            (cr - dr, cc - dc),
            (cr - dc, cc + dr),
        ]

    # Find best center: try both-integer and both-half-integer centers
    best = None  # (pairs, -size, -ncells, cr, cc)

    # Determine search range from cell positions
    all_r = [r for r, c in cells]
    all_c = [c for r, c in cells]
    min_r, max_r = min(all_r), max(all_r)
    min_c, max_c = min(all_c), max(all_c)
    span_r = max_r - min_r
    span_c = max_c - min_c
    span = max(span_r, span_c)

    # Search centers within a range that could produce valid squares
    search_r_min = max(0, min_r - span)
    search_r_max = min(rows - 1, max_r + span)
    search_c_min = max(0, min_c - span)
    search_c_max = min(cols - 1, max_c + span)

    for cr_2 in range(2 * search_r_min, 2 * search_r_max + 2):
        for cc_2 in range(2 * search_c_min, 2 * search_c_max + 2):
            # Both must be same parity (both int or both half-int)
            if (cr_2 % 2) != (cc_2 % 2):
                continue

            cr = cr_2 / 2
            cc = cc_2 / 2

            # Apply C4 to all cells, check validity
            output_cells = {}
            valid = True
            for (r, c), color in cells.items():
                for nr, nc in c4(r, c, cr, cc):
                    nr_i, nc_i = round(nr), round(nc)
                    if abs(nr - nr_i) > 0.01 or abs(nc - nc_i) > 0.01:
                        valid = False
                        break
                    if not (0 <= nr_i < rows and 0 <= nc_i < cols):
                        valid = False
                        break
                    if (nr_i, nc_i) in output_cells and output_cells[(nr_i, nc_i)] != color:
                        valid = False
                        break
                    output_cells[(nr_i, nc_i)] = color
                if not valid:
                    break

            if not valid or not output_cells:
                continue

            # Check bounding box is square with center at (cr, cc)
            out_r = [r for r, c in output_cells]
            out_c = [c for r, c in output_cells]
            omin_r, omax_r = min(out_r), max(out_r)
            omin_c, omax_c = min(out_c), max(out_c)
            h = omax_r - omin_r + 1
            w = omax_c - omin_c + 1
            if h != w:
                continue
            bb_cr = (omin_r + omax_r) / 2
            bb_cc = (omin_c + omax_c) / 2
            if abs(bb_cr - cr) > 0.01 or abs(bb_cc - cc) > 0.01:
                continue

            # Count 180° same-color pairs
            pair_count = 0
            for (r, c), color in cells.items():
                r2 = round(2 * cr - r)
                c2 = round(2 * cc - c)
                if (r2, c2) in cells and cells[(r2, c2)] == color:
                    pair_count += 1

            score = (pair_count, -h, -len(output_cells))
            if best is None or score > best[:3]:
                best = (pair_count, -h, -len(output_cells), cr, cc)

    if best is None:
        return [row[:] for row in input_grid]

    cr, cc = best[3], best[4]

    # Build output
    output = [row[:] for row in input_grid]
    for (r, c), color in cells.items():
        for nr, nc in c4(r, c, cr, cc):
            nr_i, nc_i = round(nr), round(nc)
            if 0 <= nr_i < rows and 0 <= nc_i < cols:
                output[nr_i][nc_i] = color

    return output
