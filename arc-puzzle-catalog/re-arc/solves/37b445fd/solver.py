def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = set(flat) - {bg}

    if len(non_bg) < 2:
        return [row[:] for row in input_grid]

    # Find which edge has the pattern (most non-bg cells)
    edges = {
        'top': [(input_grid[0][c], c) for c in range(cols) if input_grid[0][c] != bg],
        'bottom': [(input_grid[rows-1][c], c) for c in range(cols) if input_grid[rows-1][c] != bg],
        'left': [(input_grid[r][0], r) for r in range(rows) if input_grid[r][0] != bg],
        'right': [(input_grid[r][cols-1], r) for r in range(rows) if input_grid[r][cols-1] != bg],
    }

    best_edge = max(edges, key=lambda k: len(edges[k]))
    if len(edges[best_edge]) == 0:
        return [row[:] for row in input_grid]

    pattern_color = edges[best_edge][0][0]
    dot_color = None
    for c in non_bg:
        if c != pattern_color:
            dot_color = c
            break
    if dot_color is None:
        return [row[:] for row in input_grid]

    line_positions = sorted(set(pos for color, pos in edges[best_edge] if color == pattern_color))

    dot_set = set()
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] == dot_color:
                dot_set.add((r, c))

    output = [[bg] * cols for _ in range(rows)]

    def find_shift_direction(pos, all_pos):
        left = [p for p in all_pos if p < pos]
        right = [p for p in all_pos if p > pos]
        nearest_left = max(left) if left else None
        nearest_right = min(right) if right else None
        if nearest_left is not None and nearest_right is not None:
            if pos - nearest_left <= nearest_right - pos:
                return -1
            else:
                return 1
        elif nearest_left is not None:
            return -1
        elif nearest_right is not None:
            return 1
        return -1

    if best_edge in ('top', 'bottom'):
        row_order = list(range(rows)) if best_edge == 'top' else list(range(rows-1, -1, -1))
        active = list(line_positions)

        for idx, r in enumerate(row_order):
            deflections = {}
            for i, col in enumerate(active):
                if (r, col) in dot_set:
                    direction = find_shift_direction(col, active)
                    deflections[i] = col + direction

            for i, col in enumerate(active):
                if i in deflections:
                    new_col = deflections[i]
                    output[r][col] = dot_color
                    if 0 <= new_col < cols:
                        output[r][new_col] = pattern_color
                    if idx > 0:
                        prev_r = row_order[idx - 1]
                        if 0 <= new_col < cols:
                            output[prev_r][new_col] = pattern_color
                else:
                    output[r][col] = pattern_color

            for i, new_col in deflections.items():
                active[i] = new_col

        for r, c in dot_set:
            if output[r][c] == bg:
                output[r][c] = dot_color

    elif best_edge in ('left', 'right'):
        col_order = list(range(cols)) if best_edge == 'left' else list(range(cols-1, -1, -1))
        active = list(line_positions)

        for idx, c in enumerate(col_order):
            deflections = {}
            for i, row in enumerate(active):
                if (row, c) in dot_set:
                    direction = find_shift_direction(row, active)
                    deflections[i] = row + direction

            for i, row in enumerate(active):
                if i in deflections:
                    new_row = deflections[i]
                    output[row][c] = dot_color
                    if 0 <= new_row < rows:
                        output[new_row][c] = pattern_color
                    if idx > 0:
                        prev_c = col_order[idx - 1]
                        if 0 <= new_row < rows:
                            output[new_row][prev_c] = pattern_color
                else:
                    output[row][c] = pattern_color

            for i, new_row in deflections.items():
                active[i] = new_row

        for r, c in dot_set:
            if output[r][c] == bg:
                output[r][c] = dot_color

    return output
