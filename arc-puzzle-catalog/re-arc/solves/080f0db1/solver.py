from collections import Counter


def transform(grid):
    nrows = len(grid)
    ncols = len(grid[0])
    third = nrows // 3

    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # For each column, find the marker row and map to a color
    # based on which vertical third it falls in
    col_colors = []
    for c in range(ncols):
        marker_row = None
        for r in range(nrows):
            if grid[r][c] != bg:
                marker_row = r
                break
        if marker_row < third:
            col_colors.append(4)   # top third
        elif marker_row < 2 * third:
            col_colors.append(3)   # middle third
        else:
            col_colors.append(9)   # bottom third

    return [list(col_colors) for _ in range(nrows)]
