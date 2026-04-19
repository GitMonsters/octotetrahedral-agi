def transform(input_grid):
    """Each non-background dot emits diagonal rays that bounce off left/right
    walls and terminate at top/bottom walls."""
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    counts = Counter()
    for r in range(rows):
        for c in range(cols):
            counts[input_grid[r][c]] += 1
    bg_color = counts.most_common(1)[0][0]

    dots = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg_color:
                dots.append((r, c, input_grid[r][c]))

    if not dots:
        return [row[:] for row in input_grid]

    output = [row[:] for row in input_grid]

    for start_r, start_c, color in dots:
        # Vertical direction: from top wall go down, from bottom go up
        if start_r == 0:
            drs = [1]
        elif start_r == rows - 1:
            drs = [-1]
        else:
            drs = [1, -1]

        # Horizontal direction: from left wall go right, from right go left
        if start_c == 0:
            dcs = [1]
        elif start_c == cols - 1:
            dcs = [-1]
        else:
            dcs = [1, -1]

        for dr in drs:
            for dc in dcs:
                r, c = start_r, start_c
                cur_dc = dc
                while True:
                    output[r][c] = color
                    nr = r + dr
                    nc = c + cur_dc
                    # Stop at top/bottom walls
                    if nr < 0 or nr >= rows:
                        break
                    # Bounce off left/right walls
                    if nc < 0:
                        nc = -nc
                        cur_dc = -cur_dc
                    elif nc >= cols:
                        nc = 2 * (cols - 1) - nc
                        cur_dc = -cur_dc
                    r, c = nr, nc

    return output
