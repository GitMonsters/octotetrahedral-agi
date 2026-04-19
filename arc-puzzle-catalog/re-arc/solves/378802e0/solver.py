def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color from uniform edge rows/columns
    bg = None
    for check in [input_grid[-1], input_grid[0]]:
        if len(set(check)) == 1:
            bg = check[0]
            break
    if bg is None:
        first_col = [input_grid[r][0] for r in range(rows)]
        last_col = [input_grid[r][-1] for r in range(rows)]
        for col_vals in [last_col, first_col]:
            if len(set(col_vals)) == 1:
                bg = col_vals[0]
                break

    # Find content bounding box (skip all-bg rows/cols from edges)
    top = 0
    while top < rows and all(input_grid[top][c] == bg for c in range(cols)):
        top += 1
    bottom = rows - 1
    while bottom >= 0 and all(input_grid[bottom][c] == bg for c in range(cols)):
        bottom -= 1
    left = 0
    while left < cols and all(input_grid[r][left] == bg for r in range(rows)):
        left += 1
    right = cols - 1
    while right >= 0 and all(input_grid[r][right] == bg for r in range(rows)):
        right -= 1

    content = [[input_grid[r][c] for c in range(left, right + 1)]
               for r in range(top, bottom + 1)]
    ch, cw = len(content), len(content[0])

    # Find horizontal tile period
    tw = cw
    for p in range(1, cw + 1):
        if all(content[r][c] == content[r][c + p]
               for r in range(ch) for c in range(cw - p)):
            tw = p
            break

    # Find vertical tile period
    th = ch
    for p in range(1, ch + 1):
        if all(content[r][c] == content[r + p][c]
               for r in range(ch - p) for c in range(cw)):
            th = p
            break

    tile = [content[r][:tw] for r in range(th)]

    # Tile the entire grid with a 1-column left shift
    output = []
    for r in range(rows):
        row = []
        for c in range(cols):
            tr = (r - top) % th
            tc = (c - left + 1) % tw
            row.append(tile[tr][tc])
        output.append(row)

    return output
