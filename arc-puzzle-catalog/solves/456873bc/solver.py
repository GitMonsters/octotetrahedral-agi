def solve(grid: list[list[int]]) -> list[list[int]]:
    """Self-similar fractal pattern: the block grid arrangement mirrors the shape itself.

    Green (3) bar replaces one block row/column and must be filled per the shape template.
    Each shape at block position (r,c) gets cell (r,c) within the shape marked as 8.
    """
    H = len(grid)
    W = len(grid[0])

    # Find green bar (horizontal = full rows of 3, vertical = full cols of 3)
    green_rows = set(r for r in range(H) if all(grid[r][c] == 3 for c in range(W)))
    green_cols = set(c for c in range(W) if all(grid[r][c] == 3 for r in range(H)))
    is_horizontal = len(green_rows) > 0

    # Find separator rows (all-0 in non-green cells)
    if is_horizontal:
        sep_rows = sorted(r for r in range(H)
                          if r not in green_rows
                          and all(grid[r][c] == 0 for c in range(W)))
    else:
        sep_rows = sorted(r for r in range(H)
                          if all(grid[r][c] == 0 for c in range(W) if c not in green_cols))

    # Block row ranges from separators
    bounds = [-1] + sep_rows + [H]
    block_row_ranges = [(bounds[i] + 1, bounds[i + 1])
                        for i in range(len(bounds) - 1)
                        if bounds[i] + 1 < bounds[i + 1]]

    num_block_rows = len(block_row_ranges)
    block_height = block_row_ranges[0][1] - block_row_ranges[0][0]
    block_width = block_height + 1
    num_block_cols = (W + 1) // block_width

    block_col_ranges = [(j * block_width, min((j + 1) * block_width, W))
                        for j in range(num_block_cols)]

    # Identify which block row/col is the green bar
    if is_horizontal:
        green_bi = next((bi for bi, (rs, _) in enumerate(block_row_ranges) if rs in green_rows), None)
        green_bj = None
    else:
        green_bi = None
        green_bj = next((bj for bj, (cs, _) in enumerate(block_col_ranges) if cs in green_cols), None)

    # Extract shape template from first non-empty, non-green block
    shape = None
    for bi in range(num_block_rows):
        for bj in range(num_block_cols):
            if bi == green_bi or bj == green_bj:
                continue
            rs, re = block_row_ranges[bi]
            cs, ce = block_col_ranges[bj]
            if any(grid[r][c] == 2 for r in range(rs, re) for c in range(cs, ce)):
                shape = [[0] * block_width for _ in range(block_height)]
                for dr in range(block_height):
                    for dc in range(block_width):
                        r, c = rs + dr, cs + dc
                        if r < H and c < W:
                            shape[dr][dc] = grid[r][c]
                break
        if shape:
            break

    # Build output: place shapes where shape[bi][bj] != 0, mark diagonal with 8
    result = [[0] * W for _ in range(H)]
    for bi in range(num_block_rows):
        for bj in range(num_block_cols):
            if shape[bi][bj] != 0:
                rs, _ = block_row_ranges[bi]
                cs, _ = block_col_ranges[bj]
                for dr in range(block_height):
                    for dc in range(block_width):
                        r, c = rs + dr, cs + dc
                        if r < H and c < W:
                            result[r][c] = shape[dr][dc]
                # Mark the self-similar diagonal position with 8
                mr, mc = rs + bi, cs + bj
                if mr < H and mc < W:
                    result[mr][mc] = 8

    return result
