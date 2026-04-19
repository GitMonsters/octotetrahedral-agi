def transform(input_grid):
    """Solve ARC puzzle 846bdb03.
    
    Rule: A yellow-cornered frame defines output size with colored bars on left/right edges.
    A two-color shape elsewhere is split by color, optionally flipped if colors swap sides
    relative to the frame, then placed inside with the bar columns appended.
    """
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find yellow (4) corner cells
    yellow = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 4]
    frame_top = min(r for r, c in yellow)
    frame_bot = max(r for r, c in yellow)
    frame_left = min(c for r, c in yellow)
    frame_right = max(c for r, c in yellow)

    # Bar colors (left/right edges of frame)
    left_bar_color = right_bar_color = None
    for r in range(frame_top + 1, frame_bot):
        v = grid[r][frame_left]
        if v != 0 and v != 4:
            left_bar_color = v
            break
    for r in range(frame_top + 1, frame_bot):
        v = grid[r][frame_right]
        if v != 0 and v != 4:
            right_bar_color = v
            break

    # Frame cells to exclude from shape
    frame_cells = set()
    for r, c in yellow:
        frame_cells.add((r, c))
    for r in range(frame_top, frame_bot + 1):
        frame_cells.add((r, frame_left))
        frame_cells.add((r, frame_right))

    # Shape cells (everything else non-zero, non-yellow)
    shape_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 4 and (r, c) not in frame_cells:
                shape_cells.append((r, c, grid[r][c]))

    # Split shape by color
    colors_in_shape = set(v for r, c, v in shape_cells)
    color_a, color_b = sorted(colors_in_shape)
    cells_a = [(r, c) for r, c, v in shape_cells if v == color_a]
    cells_b = [(r, c) for r, c, v in shape_cells if v == color_b]

    # Determine left/right in shape by min column
    if min(c for r, c in cells_a) < min(c for r, c in cells_b):
        shape_left_color, shape_right_color = color_a, color_b
        shape_left_cells, shape_right_cells = cells_a, cells_b
    else:
        shape_left_color, shape_right_color = color_b, color_a
        shape_left_cells, shape_right_cells = cells_b, cells_a

    # Combined bounding box rows
    all_r = [r for r, c, v in shape_cells]
    shape_min_row = min(all_r)
    shape_height = max(all_r) - shape_min_row + 1

    # Column ranges per color
    sl_min = min(c for r, c in shape_left_cells)
    sl_w = max(c for r, c in shape_left_cells) - sl_min + 1
    sr_min = min(c for r, c in shape_right_cells)
    sr_w = max(c for r, c in shape_right_cells) - sr_min + 1

    # Extract patterns
    lp = [[0]*sl_w for _ in range(shape_height)]
    for r, c in shape_left_cells:
        lp[r - shape_min_row][c - sl_min] = shape_left_color
    rp = [[0]*sr_w for _ in range(shape_height)]
    for r, c in shape_right_cells:
        rp[r - shape_min_row][c - sr_min] = shape_right_color

    # Flip if colors swap sides between shape and frame
    if shape_left_color != left_bar_color:
        out_lp = [row[::-1] for row in rp]
        out_rp = [row[::-1] for row in lp]
    else:
        out_lp, out_rp = lp, rp

    # Add bar columns and combine
    content = []
    for i in range(shape_height):
        left_half = [left_bar_color] + out_lp[i]
        right_half = out_rp[i] + [right_bar_color]
        content.append(left_half + right_half)

    # Build output with yellow corner rows
    w = frame_right - frame_left + 1
    top = [0]*w; top[0] = 4; top[-1] = 4
    bot = [0]*w; bot[0] = 4; bot[-1] = 4
    return [top] + content + [bot]
