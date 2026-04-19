def transform(grid):
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    non_bg_colors = set(flat) - {bg}
    if len(non_bg_colors) != 2:
        return grid

    colors = list(non_bg_colors)

    # Line color: all instances in the same column on an edge
    line_color = None
    blocker_color = None
    for color in colors:
        positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        col_set = set(c for _, c in positions)
        if len(col_set) == 1:
            col_val = col_set.pop()
            if col_val == 0 or col_val == cols - 1:
                line_color = color
                blocker_color = [c2 for c2 in colors if c2 != color][0]
                break

    if line_color is None:
        return grid

    line_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == line_color]
    blocker_by_row = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == blocker_color:
                blocker_by_row[r] = c

    out = [row[:] for row in grid]

    for lr, lc in line_positions:
        if lr in blocker_by_row:
            bc = blocker_by_row[lr]
            if bc > lc:
                # Blocker to the right: fill rightward, wrap above
                for c in range(lc, bc):
                    out[lr][c] = line_color
                wrap_r = lr - 1
                if 0 <= wrap_r < rows:
                    for c in range(bc - 1, cols):
                        out[wrap_r][c] = line_color
            else:
                # Blocker to the left: fill leftward, wrap below
                for c in range(bc + 1, lc + 1):
                    out[lr][c] = line_color
                wrap_r = lr + 1
                if 0 <= wrap_r < rows:
                    for c in range(0, bc + 2):
                        out[wrap_r][c] = line_color
        else:
            # No blocker: fill entire row
            for c in range(cols):
                out[lr][c] = line_color

    return out
