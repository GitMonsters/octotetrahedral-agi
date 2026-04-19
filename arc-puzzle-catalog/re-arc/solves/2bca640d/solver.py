def transform(input_grid):
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find the template hollow rectangle (largest area wins to avoid spurious small matches)
    template = None
    max_area = 0

    for top in range(rows - 2):
        for left in range(cols - 2):
            C = grid[top][left]
            for bottom in range(top + 2, rows):
                if grid[bottom][left] != C:
                    continue
                for right in range(left + 2, cols):
                    if grid[top][right] != C or grid[bottom][right] != C:
                        continue

                    h = bottom - top + 1
                    w = right - left + 1
                    area = h * w
                    if area <= max_area:
                        continue

                    # Check full border: top and bottom rows
                    ok = True
                    for c in range(left + 1, right):
                        if grid[top][c] != C or grid[bottom][c] != C:
                            ok = False
                            break
                    if not ok:
                        continue

                    # Check left and right columns
                    for r in range(top + 1, bottom):
                        if grid[r][left] != C or grid[r][right] != C:
                            ok = False
                            break
                    if not ok:
                        continue

                    # Check interior is uniform and different from border color
                    F = grid[top + 1][left + 1]
                    if F == C:
                        continue
                    for r in range(top + 1, bottom):
                        for c in range(left + 1, right):
                            if grid[r][c] != F:
                                ok = False
                                break
                        if not ok:
                            break

                    if ok:
                        max_area = area
                        template = (top, left, bottom, right, C, F)

    if template is None:
        return grid

    top, left, bottom, right, C, F = template
    int_h = bottom - top - 1
    int_w = right - left - 1

    output = [row[:] for row in grid]

    # Find all int_h x int_w blocks of fill color F, draw rectangle border around each
    for r in range(rows - int_h + 1):
        for c in range(cols - int_w + 1):
            valid = True
            for dr in range(int_h):
                for dc in range(int_w):
                    if grid[r + dr][c + dc] != F:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                bt = r - 1       # border top
                bl = c - 1       # border left
                bb = r + int_h   # border bottom
                br = c + int_w   # border right

                if bt >= 0 and bl >= 0 and bb < rows and br < cols:
                    for bc in range(bl, br + 1):
                        output[bt][bc] = C
                        output[bb][bc] = C
                    for brow in range(bt, bb + 1):
                        output[brow][bl] = C
                        output[brow][br] = C

    return output
