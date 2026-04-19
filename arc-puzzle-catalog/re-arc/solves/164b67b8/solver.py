from collections import Counter

def transform(input_grid):
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find grid-line color: rows/cols where all cells are the same value
    grid_line_color = None
    grid_line_rows = []
    for r in range(rows):
        if len(set(grid[r])) == 1:
            grid_line_rows.append(r)
            grid_line_color = grid[r][0]

    grid_line_cols = []
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1:
            grid_line_cols.append(c)
            if grid_line_color is None:
                grid_line_color = col_vals[0]

    # Background color: most common non-grid-line value in non-grid cells
    vals = []
    for r in range(rows):
        for c in range(cols):
            if r not in grid_line_rows and c not in grid_line_cols:
                vals.append(grid[r][c])
    bg_color = Counter(vals).most_common(1)[0][0]

    # Determine block boundaries (row and column sections between grid lines)
    def sections(line_positions, total):
        secs = []
        prev = 0
        for pos in sorted(line_positions):
            if pos > prev:
                secs.append((prev, pos - 1))
            prev = pos + 1
        if prev < total:
            secs.append((prev, total - 1))
        return secs

    row_secs = sections(grid_line_rows, rows)
    col_secs = sections(grid_line_cols, cols)

    # Find yellow cell (value 4)
    yellow_r = yellow_c = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                yellow_r, yellow_c = r, c
                break
        if yellow_r is not None:
            break

    # Determine source block
    src_bi = src_bj = None
    for bi, (rs, re) in enumerate(row_secs):
        for bj, (cs, ce) in enumerate(col_secs):
            if rs <= yellow_r <= re and cs <= yellow_c <= ce:
                src_bi, src_bj = bi, bj
                break
        if src_bi is not None:
            break

    # Yellow's relative position within its block = target block index
    src_rs, src_re = row_secs[src_bi]
    src_cs, src_ce = col_secs[src_bj]
    tgt_bi = yellow_r - src_rs
    tgt_bj = yellow_c - src_cs

    # Save source block content
    block_h = src_re - src_rs + 1
    block_w = src_ce - src_cs + 1
    source_content = []
    for r in range(src_rs, src_re + 1):
        source_content.append([grid[r][c] for c in range(src_cs, src_ce + 1)])

    # Build clean output grid
    output = [[bg_color] * cols for _ in range(rows)]
    for r in grid_line_rows:
        for c in range(cols):
            output[r][c] = grid_line_color
    for c in grid_line_cols:
        for r in range(rows):
            output[r][c] = grid_line_color

    # Place pattern into target block
    tgt_rs, tgt_re = row_secs[tgt_bi]
    tgt_cs, tgt_ce = col_secs[tgt_bj]

    if (src_bi, src_bj) == (tgt_bi, tgt_bj):
        # Source IS target: copy entire block content as-is
        for ri in range(block_h):
            for ci in range(block_w):
                output[tgt_rs + ri][tgt_cs + ci] = source_content[ri][ci]
    else:
        # Source != target: copy only non-bg, non-grid-line-color cells
        for ri in range(block_h):
            for ci in range(block_w):
                val = source_content[ri][ci]
                if val != bg_color and val != grid_line_color:
                    output[tgt_rs + ri][tgt_cs + ci] = val

    return output
