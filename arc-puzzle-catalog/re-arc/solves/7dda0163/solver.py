def transform(input_grid):
    from collections import deque

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    freq = {}
    for r in range(rows):
        for c in range(cols):
            v = input_grid[r][c]
            freq[v] = freq.get(v, 0) + 1
    bg = max(freq, key=freq.get)

    # Find non-bg colors
    non_bg_colors = set(v for v in freq if v != bg)

    # Find the largest rectangular connected component across all non-bg colors
    best_rect = None  # (area, top, bottom, left, right, color)

    for color in non_bg_colors:
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if input_grid[r][c] == color and (r, c) not in visited:
                    component = []
                    queue = deque([(r, c)])
                    visited.add((r, c))
                    while queue:
                        cr, cc = queue.popleft()
                        component.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and input_grid[nr][nc] == color and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc))

                    if len(component) <= 1:
                        continue
                    rs = [p[0] for p in component]
                    cs = [p[1] for p in component]
                    top, bottom = min(rs), max(rs)
                    left, right = min(cs), max(cs)
                    area = (bottom - top + 1) * (right - left + 1)

                    if len(component) == area:
                        if best_rect is None or area > best_rect[0]:
                            best_rect = (area, top, bottom, left, right, color)

    _, rect_top, rect_bottom, rect_left, rect_right, rect_color = best_rect

    # Collect dot positions (non-bg cells outside the rectangle)
    dot_positions = []
    dot_color = None
    for r in range(rows):
        for c in range(cols):
            v = input_grid[r][c]
            if v != bg and not (rect_top <= r <= rect_bottom and rect_left <= c <= rect_right):
                dot_positions.append((r, c))
                if dot_color is None:
                    dot_color = v

    # Create output grid
    output = [[bg] * cols for _ in range(rows)]
    for r in range(rect_top, rect_bottom + 1):
        for c in range(rect_left, rect_right + 1):
            output[r][c] = rect_color

    # Frame boundaries
    ft = rect_top - 1
    fb = rect_bottom + 1
    fl = rect_left - 1
    fr = rect_right + 1

    # Project each dot onto the frame
    for r, c in dot_positions:
        if rect_top <= r <= rect_bottom:
            # In row range: project horizontally
            if c < rect_left:
                output[r][fl] = dot_color
            else:
                output[r][fr] = dot_color
        elif rect_left <= c <= rect_right:
            # In col range: project vertically
            if r < rect_top:
                output[ft][c] = dot_color
            else:
                output[fb][c] = dot_color
        else:
            # Diagonal: project to nearest corner
            target_r = ft if r < rect_top else fb
            target_c = fl if c < rect_left else fr
            output[target_r][target_c] = dot_color

    return output
