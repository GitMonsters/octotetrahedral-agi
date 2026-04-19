def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Count color frequencies
    flat = [input_grid[r][c] for r in range(rows) for c in range(cols)]
    freq = Counter(flat)
    sorted_colors = freq.most_common()

    # Top 2 by frequency are backgrounds, rest are markers
    bgs = {sorted_colors[0][0], sorted_colors[1][0]}
    markers_set = {s[0] for s in sorted_colors[2:]}

    # Find all marker positions
    marker_pos = [
        (r, c) for r in range(rows) for c in range(cols)
        if input_grid[r][c] in markers_set
    ]

    # Determine which marker color maps to which background
    # using majority vote of neighboring background cells
    marker_to_bg: dict[int, int] = {}
    for r, c in marker_pos:
        mc = input_grid[r][c]
        if mc in marker_to_bg:
            continue
        neighbor_bgs: Counter = Counter()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid[nr][nc] in bgs:
                neighbor_bgs[input_grid[nr][nc]] += 1
        if neighbor_bgs:
            marker_to_bg[mc] = neighbor_bgs.most_common(1)[0][0]

    bg_to_marker = {v: k for k, v in marker_to_bg.items()}

    # Build output: for each background cell, if any marker's diagonal
    # passes through it, replace with that region's marker color
    output = [row[:] for row in input_grid]
    for r in range(rows):
        for c in range(cols):
            val = input_grid[r][c]
            if val in bgs and val in bg_to_marker:
                mk = bg_to_marker[val]
                for mr, mc in marker_pos:
                    if abs(r - mr) == abs(c - mc) and r != mr:
                        output[r][c] = mk
                        break

    return output
