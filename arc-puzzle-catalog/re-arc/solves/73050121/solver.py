def transform(grid):
    """
    Pattern: There's a rectangular frame (box) containing a small pattern of a 'special' color.
    All connected components of the 'scattered' color whose shape matches that pattern
    get recolored to the special color. If no frame exists, use the smallest cluster
    of the least-frequent non-background color as the reference shape.
    """
    from collections import Counter, deque

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Background = most frequent color
    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1
    bg = color_counts.most_common(1)[0][0]
    non_bg_colors = [c for c in color_counts if c != bg]

    def find_components(color):
        visited = set()
        components = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r, c) not in visited:
                    comp = []
                    q = deque([(r, c)])
                    visited.add((r, c))
                    while q:
                        cr, cc = q.popleft()
                        comp.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == color:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                    components.append(sorted(comp))
        return components

    def shape_sig(cells):
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        return tuple(sorted((r - min_r, c - min_c) for r, c in cells))

    # Find rectangular frames
    def find_frames():
        found = []
        for color in non_bg_colors:
            for r1 in range(rows - 2):
                for r2 in range(r1 + 2, rows):
                    for c1 in range(cols - 2):
                        for c2 in range(c1 + 2, cols):
                            if grid[r1][c1] != color or grid[r1][c2] != color:
                                continue
                            if grid[r2][c1] != color or grid[r2][c2] != color:
                                continue
                            if not all(grid[r1][c] == color for c in range(c1, c2 + 1)):
                                continue
                            if not all(grid[r2][c] == color for c in range(c1, c2 + 1)):
                                continue
                            if not all(grid[r][c1] == color for r in range(r1, r2 + 1)):
                                continue
                            if not all(grid[r][c2] == color for r in range(r1, r2 + 1)):
                                continue
                            interior_colors = set()
                            for r in range(r1 + 1, r2):
                                for c in range(c1 + 1, c2):
                                    v = grid[r][c]
                                    if v != color and v != bg:
                                        interior_colors.add(v)
                            if interior_colors:
                                found.append((r1, c1, r2, c2, color, interior_colors))
        return found

    frames = find_frames()

    if frames:
        r1, c1, r2, c2, frame_color, interior_colors = frames[0]
        special_color = list(interior_colors)[0]

        interior_cells = []
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if grid[r][c] == special_color:
                    interior_cells.append((r, c))

        target_shape = shape_sig(interior_cells)

        scattered_color = None
        for c in non_bg_colors:
            if c != frame_color and c != special_color:
                scattered_color = c
                break

        if scattered_color is not None:
            comps = find_components(scattered_color)
            for comp in comps:
                if shape_sig(comp) == target_shape:
                    for r, c in comp:
                        result[r][c] = special_color
    else:
        # No frame: special color has fewer total cells
        non_bg_sorted = sorted(non_bg_colors, key=lambda c: color_counts[c])
        special_color = non_bg_sorted[0]
        scattered_color = non_bg_sorted[1]

        special_comps = find_components(special_color)
        if special_comps:
            smallest = min(special_comps, key=len)
            target_shape = shape_sig(smallest)

            scattered_comps = find_components(scattered_color)
            for comp in scattered_comps:
                if shape_sig(comp) == target_shape:
                    for r, c in comp:
                        result[r][c] = special_color

    return result
