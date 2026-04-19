from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find all non-background pixels
    non_bg = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg.append((r, c, grid[r][c]))

    # Check each edge for markers
    left_edge = [(r, c, v) for r, c, v in non_bg if c == 0]
    right_edge = [(r, c, v) for r, c, v in non_bg if c == cols - 1]
    top_edge = [(r, c, v) for r, c, v in non_bg if r == 0]
    bottom_edge = [(r, c, v) for r, c, v in non_bg if r == rows - 1]

    edges = {'left': left_edge, 'right': right_edge, 'top': top_edge, 'bottom': bottom_edge}
    edge_name = max(edges, key=lambda k: len(edges[k]))
    edge_markers = edges[edge_name]

    if not edge_markers:
        return [row[:] for row in grid]

    edge_color = edge_markers[0][2]

    # Identify interior pixels (not on the marker edge)
    interior = []
    for r, c, v in non_bg:
        on_marker_edge = False
        if edge_name == 'left' and c == 0: on_marker_edge = True
        elif edge_name == 'right' and c == cols - 1: on_marker_edge = True
        elif edge_name == 'top' and r == 0: on_marker_edge = True
        elif edge_name == 'bottom' and r == rows - 1: on_marker_edge = True
        if not on_marker_edge:
            interior.append((r, c, v))

    output = [[bg for _ in range(cols)] for _ in range(rows)]

    if edge_name in ('left', 'right'):
        line_rows = {r for r, c, v in edge_markers}

        deflectors = {}
        dots = []
        for r, c, v in interior:
            if r in line_rows and v == edge_color:
                deflectors.setdefault(r, []).append((r, c, v))
            else:
                dots.append((r, c, v))

        for r in line_rows:
            if r not in deflectors:
                for c in range(cols):
                    output[r][c] = edge_color
            else:
                dr, dc, dv = deflectors[r][0]
                if edge_name == 'left':
                    for c in range(0, dc + 1):
                        output[r][c] = edge_color
                    shift_r = r - 1
                    if shift_r >= 0:
                        for c in range(max(0, dc - 1), cols):
                            output[shift_r][c] = edge_color
                elif edge_name == 'right':
                    for c in range(dc, cols):
                        output[r][c] = edge_color
                    shift_r = r - 1
                    if shift_r >= 0:
                        for c in range(0, min(cols, dc + 2)):
                            output[shift_r][c] = edge_color

        for r, c, v in dots:
            output[r][c] = v

    elif edge_name in ('top', 'bottom'):
        line_cols = {c for r, c, v in edge_markers}

        deflectors = {}
        dots = []
        for r, c, v in interior:
            if c in line_cols and v == edge_color:
                deflectors.setdefault(c, []).append((r, c, v))
            else:
                dots.append((r, c, v))

        for c in line_cols:
            if c not in deflectors:
                for r in range(rows):
                    output[r][c] = edge_color
            else:
                dr, dc, dv = deflectors[c][0]
                if edge_name == 'bottom':
                    for r in range(dr, rows):
                        output[r][c] = edge_color
                    shift_c = c - 1
                    if shift_c >= 0:
                        for r in range(0, dr + 2):
                            output[r][shift_c] = edge_color
                elif edge_name == 'top':
                    for r in range(0, dr + 1):
                        output[r][c] = edge_color
                    shift_c = c - 1
                    if shift_c >= 0:
                        for r in range(max(0, dr - 1), rows):
                            output[r][shift_c] = edge_color

        for r, c, v in dots:
            output[r][c] = v

    return output
