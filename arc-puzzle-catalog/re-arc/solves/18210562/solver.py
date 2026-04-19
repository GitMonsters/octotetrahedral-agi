def transform(input_grid):
    from collections import Counter

    R = len(input_grid)
    C = len(input_grid[0])

    # Background = most common color
    bg = Counter(v for row in input_grid for v in row).most_common(1)[0][0]

    # Find cross: horizontal line (full row of non-bg color)
    cross_color = None
    h_row = None
    for r in range(R):
        if len(set(input_grid[r])) == 1 and input_grid[r][0] != bg:
            h_row = r
            cross_color = input_grid[r][0]
            break

    # Find vertical line (column where all cells are cross_color)
    v_col = None
    for c in range(C):
        if all(input_grid[r][c] == cross_color for r in range(R)):
            v_col = c
            break

    # Find noise cells (non-bg, not on cross lines)
    noise = []
    for r in range(R):
        for c in range(C):
            if r == h_row or c == v_col:
                continue
            if input_grid[r][c] != bg:
                noise.append((r, c))

    N = len(noise)
    top = h_row
    bottom = R - 1 - h_row
    left = v_col
    right = C - 1 - v_col

    if N > 0:
        # Shift by N away from the noisy quadrant
        nr, nc = noise[0]
        if nr < h_row and nc < v_col:      # TL noise -> down-right
            dr, dc = +N, +N
        elif nr < h_row and nc > v_col:     # TR noise -> down-left
            dr, dc = +N, -N
        elif nr > h_row and nc < v_col:     # BL noise -> up-right
            dr, dc = -N, +N
        else:                                # BR noise -> up-left
            dr, dc = -N, -N
    else:
        # No noise: find the square quadrant and grow it
        if top == right:       # TR square -> grow down-left
            s = min(bottom, left) // 2
            dr, dc = +s, -s
        elif top == left:      # TL square -> grow down-right
            s = min(bottom, right) // 2
            dr, dc = +s, +s
        elif bottom == left:   # BL square -> grow up-right
            s = min(top, right) // 2
            dr, dc = -s, +s
        elif bottom == right:  # BR square -> grow up-left
            s = min(top, left) // 2
            dr, dc = -s, -s
        else:
            dr, dc = 0, 0

    new_h = h_row + dr
    new_v = v_col + dc

    # Build clean output with new cross position
    output = [[bg] * C for _ in range(R)]
    for c in range(C):
        output[new_h][c] = cross_color
    for r in range(R):
        output[r][new_v] = cross_color
    return output
