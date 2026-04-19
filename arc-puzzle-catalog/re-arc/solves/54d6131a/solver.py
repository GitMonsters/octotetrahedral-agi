from collections import Counter


def transform(grid):
    R, C = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    rows_nz = [r for r in range(R) if any(grid[r][c] != bg for c in range(C))]
    cols_nz = [c for c in range(C) if any(grid[r][c] != bg for r in range(R))]
    if not rows_nz:
        return [row[:] for row in grid]

    rmin, rmax = min(rows_nz), max(rows_nz)
    cmin, cmax = min(cols_nz), max(cols_nz)
    top = rmin
    bottom = R - 1 - rmax
    left_sp = cmin
    right_sp = C - 1 - cmax

    colors = sorted(set(v for row in grid for v in row) - {bg})
    out = [row[:] for row in grid]

    if len(colors) == 1:
        clr = colors[0]
        # Row sigs: which cols have clr per row
        row_cols = {}
        for r in range(rmin, rmax + 1):
            row_cols[r] = sorted(c for c in range(cmin, cmax + 1) if grid[r][c] == clr)
        # Col sigs: which rows have clr per col
        col_rows = {}
        for c in range(cmin, cmax + 1):
            col_rows[c] = sorted(r for r in range(rmin, rmax + 1) if grid[r][c] == clr)

        std_row = Counter(tuple(row_cols[r]) for r in range(rmin, rmax + 1) if row_cols[r]).most_common(1)[0][0]
        std_col = Counter(tuple(col_rows[c]) for c in range(cmin, cmax + 1) if col_rows[c]).most_common(1)[0][0]
        std_c_min, std_c_max = min(std_row), max(std_row)
        std_r_min, std_r_max = min(std_col), max(std_col)

        # Irregular cols: clr appears in rows OUTSIDE standard row range
        irreg_cols = set()
        for r in range(rmin, rmax + 1):
            if r < std_r_min or r > std_r_max:
                for c in row_cols[r]:
                    irreg_cols.add(c)

        # Irregular rows: clr extends OUTSIDE standard col range
        irreg_rows = [r for r in range(std_r_min, std_r_max + 1)
                      if row_cols[r] and (min(row_cols[r]) < std_c_min or max(row_cols[r]) > std_c_max)]

        if not irreg_cols and not irreg_rows:
            # Uniform rectangle → formula-based projection
            N_rows = left_sp   # number of full-width rows = left space
            N_cols = right_sp  # number of col projections = right space
            H_block = rmax - rmin + 1
            W_block = cmax - cmin + 1

            # Interior size: exclude rows/cols at free (non-boundary) edges
            free_top = 1 if top > 0 else 0
            free_bottom = 1 if bottom > 0 else 0
            free_left = 1 if left_sp > 0 else 0
            free_right = 1 if right_sp > 0 else 0
            D_r = H_block - free_top - free_bottom
            D_c = W_block - free_left - free_right

            def formula_positions(D, N):
                if N <= 0 or D <= 1:
                    return []
                return [(2 * k - 1) * (D - 1) // (2 * N) for k in range(1, N + 1)]

            interior_row_start = rmin + free_top
            interior_col_start = cmin + free_left

            proj_rows = [interior_row_start + p for p in formula_positions(D_r, N_rows)]
            proj_cols = [interior_col_start + p for p in formula_positions(D_c, N_cols)]

            # Projected rows: extend to full width
            for r in proj_rows:
                for c in range(C):
                    out[r][c] = clr
            # Projected cols: appear in vertical spaces
            if top > 0:
                for c in proj_cols:
                    for r in range(0, rmin):
                        out[r][c] = clr
            if bottom > 0:
                for c in proj_cols:
                    for r in range(rmax + 1, R):
                        out[r][c] = clr
            return out

        else:
            # Blob with protrusions
            # Projection direction for rows (horizontal)
            if right_sp >= left_sp:
                row_proj_start = cmax + left_sp + 1
                row_proj_end = C - 1
            else:
                row_proj_end = cmin - right_sp - 1
                row_proj_start = 0

            # Projection direction for cols (vertical)
            if top >= bottom:
                col_proj_start = 0
                col_proj_end = rmin - bottom - 1
            else:
                col_proj_start = rmax + top + 1
                col_proj_end = R - 1

            # Build output from scratch (all bg)
            for r in range(R):
                for c in range(C):
                    out[r][c] = bg

            # Place irregular col projections
            for r in range(col_proj_start, col_proj_end + 1):
                for c in irreg_cols:
                    out[r][c] = clr

            # Place irregular row projections
            for r in irreg_rows:
                for c in range(row_proj_start, row_proj_end + 1):
                    out[r][c] = clr

            # Place standard rows (in std row range, not irregular rows)
            std_cols_kept = [c for c in std_row if c not in irreg_cols]
            for r in range(std_r_min, std_r_max + 1):
                if r not in irreg_rows:
                    for c in std_cols_kept:
                        out[r][c] = clr
            return out

    elif len(colors) == 2:
        # Frame structure: border_clr and interior_clr
        border_clr = grid[rmin][cmin]
        interior_clr = [c for c in colors if c != border_clr][0]

        # Interior cells per row
        row_int = {}
        for r in range(rmin, rmax + 1):
            ic = sorted(c for c in range(cmin, cmax + 1) if grid[r][c] == interior_clr)
            row_int[r] = ic

        # Interior cells per col
        col_int = {}
        for c in range(cmin, cmax + 1):
            ir = sorted(r for r in range(rmin, rmax + 1) if grid[r][c] == interior_clr)
            col_int[c] = ir

        # Standard interior row sig (most common)
        std_int_row = Counter(tuple(row_int[r]) for r in range(rmin, rmax + 1) if row_int[r]).most_common(1)[0][0]
        std_int_col = Counter(tuple(col_int[c]) for c in range(cmin, cmax + 1) if col_int[c]).most_common(1)[0][0]

        std_int_c_min = min(std_int_row) if std_int_row else cmin
        std_int_c_max = max(std_int_row) if std_int_row else cmax
        std_int_r_min = min(std_int_col) if std_int_col else rmin
        std_int_r_max = max(std_int_col) if std_int_col else rmax

        # Irregular interior cols: interior appears OUTSIDE standard interior row range
        irreg_int_cols = set()
        for c in range(cmin, cmax + 1):
            for r in col_int[c]:
                if r < std_int_r_min or r > std_int_r_max:
                    irreg_int_cols.add(c)

        # Irregular interior rows: interior extends OUTSIDE standard interior col range
        irreg_int_rows = [r for r in range(rmin, rmax + 1)
                          if row_int[r] and (min(row_int[r]) < std_int_c_min or max(row_int[r]) > std_int_c_max)]

        # Build output
        # Start with copy (frame structure stays)
        # 1. Irregular rows: fill bbox portion with border_clr, project interior_clr to ALL bg space
        for r in irreg_int_rows:
            for c in range(C):
                if cmin <= c <= cmax:
                    out[r][c] = border_clr
                else:
                    out[r][c] = interior_clr

        # 2. Irregular cols: project interior_clr above and below bbox
        for c in irreg_int_cols:
            for r in range(0, rmin):
                out[r][c] = interior_clr
            for r in range(rmax + 1, R):
                out[r][c] = interior_clr
            # Within bbox: fill holes in border rows with border_clr
            for r in range(rmin, rmax + 1):
                if r not in irreg_int_rows:
                    if grid[r][c] == interior_clr and (r < std_int_r_min or r > std_int_r_max):
                        out[r][c] = border_clr

        # 3. Standard rows: trim interior to standard range (remove irreg col positions)
        for r in range(rmin, rmax + 1):
            if r in irreg_int_rows:
                continue
            if not row_int[r]:
                continue  # border row, no interior
            for c in irreg_int_cols:
                if cmin <= c <= cmax:
                    out[r][c] = border_clr

        return out

    return out
