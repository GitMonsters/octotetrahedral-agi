def solve(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    # Find the 7x7 template of 1s (with colored markers inside)
    ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    tr0, tr1 = min(r for r, c in ones), max(r for r, c in ones)
    tc0, tc1 = min(c for r, c in ones), max(c for r, c in ones)

    # Extract the 4 marker colors from template at fixed relative positions
    # Markers are at (1,1), (1,5), (5,1), (5,5) relative to template origin
    quadrant_map = {
        (1, 1): (0, 0),  # top-left quadrant: rows 0-2, cols 0-2
        (1, 5): (0, 4),  # top-right quadrant: rows 0-2, cols 4-6
        (5, 1): (4, 0),  # bottom-left quadrant: rows 4-6, cols 0-2
        (5, 5): (4, 4),  # bottom-right quadrant: rows 4-6, cols 4-6
    }

    template_cells = set()
    for r in range(tr0, tr1 + 1):
        for c in range(tc0, tc1 + 1):
            template_cells.add((r, c))

    markers = {}  # color -> (quadrant_row_start, quadrant_col_start)
    for (dr, dc), qstart in quadrant_map.items():
        color = grid[tr0 + dr][tc0 + dc]
        markers[color] = qstart

    # For each marker color, find its shape outside the template
    output = [[0] * 7 for _ in range(7)]

    for color, (qr, qc) in markers.items():
        # Collect all cells of this color NOT in the template
        cells = [
            (r, c) for r in range(rows) for c in range(cols)
            if grid[r][c] == color and (r, c) not in template_cells
        ]
        sr0 = min(r for r, c in cells)
        sc0 = min(c for r, c in cells)

        # Place the 3x3 shape into the quadrant
        for r, c in cells:
            output[qr + (r - sr0)][qc + (c - sc0)] = color

    return output
