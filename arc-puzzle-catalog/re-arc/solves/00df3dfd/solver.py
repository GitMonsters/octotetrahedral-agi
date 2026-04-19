def transform(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])

    correct = [row[:] for row in input_grid]
    damaged_cells = []

    # Use T-B symmetry to find damage
    for r in range(H // 2):
        mirror_r = H - 1 - r
        cols = [c for c in range(W) if input_grid[r][c] != input_grid[mirror_r][c]]
        if not cols:
            continue

        # Determine which row is damaged: the damaged row has uniform values
        r_vals = set(input_grid[r][c] for c in cols)
        m_vals = set(input_grid[mirror_r][c] for c in cols)

        if len(m_vals) == 1 and len(r_vals) > 1:
            # mirror_r is damaged
            for c in cols:
                correct[mirror_r][c] = input_grid[r][c]
                damaged_cells.append((mirror_r, c))
        elif len(r_vals) == 1 and len(m_vals) > 1:
            # r is damaged
            for c in cols:
                correct[r][c] = input_grid[mirror_r][c]
                damaged_cells.append((r, c))
        elif len(r_vals) == 1 and len(m_vals) == 1:
            # Both uniform — use L-R symmetry to break tie
            r_lr = all(input_grid[r][c] == input_grid[r][W - 1 - c] for c in cols)
            m_lr = all(input_grid[mirror_r][c] == input_grid[mirror_r][W - 1 - c] for c in cols)
            if m_lr and not r_lr:
                for c in cols:
                    correct[r][c] = input_grid[mirror_r][c]
                    damaged_cells.append((r, c))
            else:
                for c in cols:
                    correct[mirror_r][c] = input_grid[r][c]
                    damaged_cells.append((mirror_r, c))

    if not damaged_cells:
        return input_grid

    min_r = min(r for r, c in damaged_cells)
    max_r = max(r for r, c in damaged_cells)
    min_c = min(c for r, c in damaged_cells)
    max_c = max(c for r, c in damaged_cells)

    output = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(correct[r][c])
        output.append(row)

    return output
