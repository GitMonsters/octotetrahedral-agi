def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find the 2x2 block of 6s
    six_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 6]
    six_row_min = min(r for r, c in six_positions)
    six_row_max = max(r for r, c in six_positions)
    six_col_min = min(c for r, c in six_positions)
    six_col_max = max(c for r, c in six_positions)

    # Find all-0 rows and columns (separators + borders)
    zero_rows = sorted(r for r in range(rows) if all(grid[r][c] == 0 for c in range(cols)))
    zero_cols = sorted(c for c in range(cols) if all(grid[r][c] == 0 for r in range(rows)))

    has_row_blocks = len(zero_rows) > 2
    has_col_blocks = len(zero_cols) > 2

    result = [row[:] for row in grid]

    if has_col_blocks and not has_row_blocks:
        # Column blocks, continuous rows
        col_blocks = []
        for i in range(len(zero_cols) - 1):
            s, e = zero_cols[i] + 1, zero_cols[i + 1]
            if s < e:
                col_blocks.append((s, e))

        col_to_block = {}
        for idx, (s, e) in enumerate(col_blocks):
            for c in range(s, e):
                col_to_block[c] = idx

        six_block = col_to_block[six_col_min]

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 8:
                    continue
                bidx = col_to_block.get(c)
                if bidx is None:
                    continue
                bdist = abs(bidx - six_block)
                if bdist == 0:
                    continue
                if r < six_row_min:
                    cdist = six_row_min - r
                elif r > six_row_max:
                    cdist = r - six_row_max
                else:
                    cdist = 0
                if cdist >= 2 * bdist - 1:
                    result[r][c] = 4

    elif has_row_blocks and not has_col_blocks:
        # Row blocks, continuous columns
        row_blocks = []
        for i in range(len(zero_rows) - 1):
            s, e = zero_rows[i] + 1, zero_rows[i + 1]
            if s < e:
                row_blocks.append((s, e))

        row_to_block = {}
        for idx, (s, e) in enumerate(row_blocks):
            for r in range(s, e):
                row_to_block[r] = idx

        six_block = row_to_block[six_row_min]

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 8:
                    continue
                bidx = row_to_block.get(r)
                if bidx is None:
                    continue
                bdist = abs(bidx - six_block)
                if bdist == 0:
                    continue
                if c < six_col_min:
                    cdist = six_col_min - c
                elif c > six_col_max:
                    cdist = c - six_col_max
                else:
                    cdist = 0
                if cdist >= 2 * bdist - 1:
                    result[r][c] = 4

    return result
