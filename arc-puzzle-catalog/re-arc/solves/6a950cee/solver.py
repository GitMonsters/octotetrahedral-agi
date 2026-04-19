def transform(input_grid):
    import numpy as np
    from collections import Counter

    grid = np.array(input_grid)
    H, W = grid.shape

    # Find background (most common color)
    bg = Counter(grid.flatten()).most_common(1)[0][0]
    non_bg = sorted(set(grid.flatten()) - {bg})

    # For each non-bg color, check if it forms exactly 2 parallel line segments
    def find_line_color(grid, bg, non_bg_colors):
        for c in non_bg_colors:
            positions = list(zip(*np.where(grid == c)))
            if len(positions) < 4:
                continue

            # Check vertical lines
            col_groups = {}
            for r, cc in positions:
                col_groups.setdefault(int(cc), []).append(int(r))

            vertical_lines = []
            for col, rs in sorted(col_groups.items()):
                rs_sorted = sorted(rs)
                if len(rs_sorted) >= 2 and rs_sorted == list(range(rs_sorted[0], rs_sorted[-1] + 1)):
                    vertical_lines.append((col, rs_sorted[0], rs_sorted[-1]))

            if len(vertical_lines) == 2:
                total_in_lines = sum(vl[2] - vl[1] + 1 for vl in vertical_lines)
                if total_in_lines == len(positions):
                    return c, 'V', vertical_lines

            # Check horizontal lines
            row_groups = {}
            for r, cc in positions:
                row_groups.setdefault(int(r), []).append(int(cc))

            horizontal_lines = []
            for row, cs in sorted(row_groups.items()):
                cs_sorted = sorted(cs)
                if len(cs_sorted) >= 2 and cs_sorted == list(range(cs_sorted[0], cs_sorted[-1] + 1)):
                    horizontal_lines.append((row, cs_sorted[0], cs_sorted[-1]))

            if len(horizontal_lines) == 2:
                total_in_lines = sum(hl[2] - hl[1] + 1 for hl in horizontal_lines)
                if total_in_lines == len(positions):
                    return c, 'H', horizontal_lines

        return None, None, None

    line_color, orientation, lines = find_line_color(grid, bg, non_bg)

    if line_color is not None:
        # 3-color case: extract rectangle from line segments
        if orientation == 'V':
            c1, c2 = lines[0][0], lines[1][0]
            r_min = lines[0][1] - 1
            r_max = lines[0][2] + 1
            c_min, c_max = min(c1, c2), max(c1, c2)
        else:  # 'H'
            r1, r2 = lines[0][0], lines[1][0]
            c_min = lines[0][1] - 1
            c_max = lines[0][2] + 1
            r_min, r_max = min(r1, r2), max(r1, r2)

        result = grid[r_min:r_max + 1, c_min:c_max + 1]
        return result.tolist()

    # 2-color case: find rectangle by corners
    # Find 4 non-bg cells forming rectangle corners with cleanest sides
    color = non_bg[0]
    positions = set(zip(*np.where(grid == color)))
    pos_by_row = {}
    for r, c in positions:
        pos_by_row.setdefault(int(r), []).append(int(c))

    best = None
    best_score = (-1, -1)  # (border_density, -strays)

    rows_list = sorted(pos_by_row.keys())
    for i, r1 in enumerate(rows_list):
        for r2 in rows_list[i + 1:]:
            # Find common columns where both rows have the non-bg color
            cols_r1 = set(pos_by_row[r1])
            cols_r2 = set(pos_by_row[r2])
            common_cols = sorted(cols_r1 & cols_r2)

            for ci, c1 in enumerate(common_cols):
                for c2 in common_cols[ci + 1:]:
                    v_side_len = r2 - r1 - 1
                    h_side_len = c2 - c1 - 1

                    if min(v_side_len, h_side_len) < 3:
                        continue

                    # Check if at least one pair of sides is completely clean
                    v_clean = True
                    if v_side_len > 0:
                        for r in range(r1 + 1, r2):
                            if grid[r][c1] != bg or grid[r][c2] != bg:
                                v_clean = False
                                break

                    h_clean = True
                    if h_side_len > 0:
                        for c in range(c1 + 1, c2):
                            if grid[r1][c] != bg or grid[r2][c] != bg:
                                h_clean = False
                                break

                    if not v_clean and not h_clean:
                        continue

                    # Count non-bg on all perimeter (excluding corners)
                    perimeter_non_bg = 0
                    perimeter_len = 0
                    # V sides
                    for r in range(r1 + 1, r2):
                        perimeter_len += 2
                        if grid[r][c1] != bg:
                            perimeter_non_bg += 1
                        if grid[r][c2] != bg:
                            perimeter_non_bg += 1
                    # H sides
                    for c in range(c1 + 1, c2):
                        perimeter_len += 2
                        if grid[r1][c] != bg:
                            perimeter_non_bg += 1
                        if grid[r2][c] != bg:
                            perimeter_non_bg += 1

                    # Include corners in density calculation
                    total_border = perimeter_non_bg + 4  # 4 corners
                    total_perimeter = perimeter_len + 4
                    density = total_border / total_perimeter if total_perimeter > 0 else 0

                    score = (density, -(perimeter_non_bg - 0))  # higher density, fewer strays

                    if score > best_score:
                        best_score = score
                        best = (r1, r2, c1, c2)

    if best:
        r1, r2, c1, c2 = best
        result = grid[r1:r2 + 1, c1:c2 + 1]
        return result.tolist()

    return input_grid
