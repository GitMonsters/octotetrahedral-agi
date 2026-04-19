from collections import defaultdict, Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Background value (most common)
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg_val = Counter(flat).most_common(1)[0][0]

    # Find separator structure
    sep_val, block_h, block_w, sep_rows, sep_cols = _find_separators(grid, rows, cols, bg_val)

    # Block start positions
    row_starts = [r for r in range(rows) if r not in sep_rows and (r == 0 or r - 1 in sep_rows)]
    col_starts = [c for c in range(cols) if c not in sep_cols and (c == 0 or c - 1 in sep_cols)]

    # Extract non-bg/non-sep markers grouped by (value, local_row, local_col)
    groups = defaultdict(set)
    for br, rs in enumerate(row_starts):
        for bc, cs in enumerate(col_starts):
            for lr in range(block_h):
                for lc in range(block_w):
                    r, c = rs + lr, cs + lc
                    if r < rows and c < cols:
                        v = grid[r][c]
                        if v != bg_val and v != sep_val:
                            groups[(v, lr, lc)].add((br, bc))

    # Fill between extremes along block-rows and block-columns
    filled = {}
    for key, blocks in groups.items():
        result = set(blocks)
        by_row = defaultdict(list)
        by_col = defaultdict(list)
        for br, bc in blocks:
            by_row[br].append(bc)
            by_col[bc].append(br)
        for br, bcs in by_row.items():
            for bc in range(min(bcs), max(bcs) + 1):
                result.add((br, bc))
        for bc, brs in by_col.items():
            for br in range(min(brs), max(brs) + 1):
                result.add((br, bc))
        filled[key] = result

    # Construct output
    output = [row[:] for row in grid]
    for (v, lr, lc), blocks in filled.items():
        for br, bc in blocks:
            r = row_starts[br] + lr
            c = col_starts[bc] + lc
            if r < rows and c < cols:
                output[r][c] = v
    return output


def _find_separators(grid, rows, cols, bg_val):
    # Rows/cols where all cells share the same value
    same_rows = defaultdict(set)
    for r in range(rows):
        if len(set(grid[r])) == 1:
            same_rows[grid[r][0]].add(r)

    same_cols = defaultdict(set)
    for c in range(cols):
        if len(set(grid[r][c] for r in range(rows))) == 1:
            same_cols[grid[0][c]].add(c)

    candidates = []
    for v in set(same_rows) & set(same_cols):
        sr, sc = same_rows[v], same_cols[v]

        # Smallest block_h where separator rows at bh, 2*bh+1, ... tile perfectly
        bh = _smallest_block(sr, rows)
        if bh is None:
            continue
        bw = _smallest_block(sc, cols)
        if bw is None:
            continue
        candidates.append((v, bh, bw))

    if not candidates:
        return None
    # Prefer explicit separator (!=bg), then smallest blocks
    candidates.sort(key=lambda x: (x[0] == bg_val, x[1] * x[2]))
    v, bh, bw = candidates[0]
    return v, bh, bw, set(range(bh, rows, bh + 1)), set(range(bw, cols, bw + 1))


def _smallest_block(same_set, dim):
    for b in range(1, dim):
        expected = set(range(b, dim, b + 1))
        if expected and expected.issubset(same_set):
            n = len(expected) + 1
            if n * b + len(expected) == dim:
                return b
    return None
