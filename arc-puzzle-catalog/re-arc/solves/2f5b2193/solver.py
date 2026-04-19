def transform(input_grid):
    from collections import Counter

    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Count cells per color
    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1

    # Background = most common color
    bg = color_counts.most_common(1)[0][0]

    # Marker colors have exactly 2 cells each
    marker_colors = sorted([color for color, count in color_counts.items() if count == 2])

    if len(marker_colors) < 2:
        return grid

    # Line color (drawn) and endpoint color
    line_color = marker_colors[0]  # typically 2
    end_color = marker_colors[1]   # typically 3

    # Find marker cells
    line_cells = []
    end_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == line_color:
                line_cells.append((r, c))
            elif grid[r][c] == end_color:
                end_cells.append((r, c))

    # Endpoint center
    end_r = sum(r for r, c in end_cells) / len(end_cells)
    end_c = sum(c for r, c in end_cells) / len(end_cells)

    # Determine line pair orientation and initial direction
    if line_cells[0][0] == line_cells[1][0]:  # horizontal pair
        row = line_cells[0][0]
        c_min = min(c for _, c in line_cells)
        c_max = max(c for _, c in line_cells)
        if end_c > c_max:
            dr, dc = 0, 1
            pos = (row, c_max)
        else:
            dr, dc = 0, -1
            pos = (row, c_min)
    else:  # vertical pair
        col = line_cells[0][1]
        r_min = min(r for r, _ in line_cells)
        r_max = max(r for r, _ in line_cells)
        if end_r < r_min:
            dr, dc = -1, 0
            pos = (r_min, col)
        else:
            dr, dc = 1, 0
            pos = (r_max, col)

    # Shoot line segments, turning toward endpoint at obstacles
    for _ in range(10):
        r, c = pos
        moved = False
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                break
            if grid[nr][nc] != bg:
                break
            r, c = nr, nc
            grid[r][c] = line_color
            moved = True
        pos = (r, c)

        # Check if adjacent to endpoint
        for er, ec in end_cells:
            if abs(r - er) + abs(c - ec) <= 1:
                return grid

        if not moved:
            break

        # Turn toward endpoint
        if dr == 0:  # was horizontal, turn vertical
            dr, dc = (-1, 0) if end_r < r else (1, 0)
        else:  # was vertical, turn horizontal
            dr, dc = (0, -1) if end_c < c else (0, 1)

    return grid
