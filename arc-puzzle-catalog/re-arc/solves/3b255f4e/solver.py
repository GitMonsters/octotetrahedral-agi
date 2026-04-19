from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = set(flat) - {bg}

    # Find non-bg divider rows and columns (color filling >= n-2 cells)
    row_divs = []
    col_divs = []
    for color in non_bg:
        for r in range(rows):
            if sum(1 for c in range(cols) if grid[r][c] == color) >= cols - 2:
                row_divs.append((r, color))
        for c in range(cols):
            if sum(1 for r in range(rows) if grid[r][c] == color) >= rows - 2:
                col_divs.append((c, color))

    # bg-colored row dividers (entire row is bg, overrides column dividers)
    if len(row_divs) < 2:
        existing = {rd[0] for rd in row_divs}
        for r in range(rows):
            if r not in existing and all(grid[r][c] == bg for c in range(cols)):
                row_divs.append((r, bg))

    # bg-colored column dividers (strict: entire column is bg)
    if len(col_divs) < 2:
        existing = {cd[0] for cd in col_divs}
        for c in range(cols):
            if c not in existing and all(grid[r][c] == bg for r in range(rows)):
                col_divs.append((c, bg))

    # Relaxed bg column dividers: exclude row divider intersections
    if len(col_divs) < 2:
        div_row_set = {rd[0] for rd in row_divs}
        existing = {cd[0] for cd in col_divs}
        for c in range(cols):
            if c not in existing:
                if all(grid[r][c] == bg for r in range(rows) if r not in div_row_set):
                    col_divs.append((c, bg))

    # Relaxed bg row dividers: exclude column divider intersections
    if len(row_divs) < 2:
        div_col_set = {cd[0] for cd in col_divs}
        existing = {rd[0] for rd in row_divs}
        for r in range(rows):
            if r not in existing:
                if all(grid[r][c] == bg for c in range(cols) if c not in div_col_set):
                    row_divs.append((r, bg))

    row_divs.sort()
    col_divs.sort()
    r1, r1c = row_divs[0]
    r2, r2c = row_divs[1]
    c1, c1c = col_divs[0]
    c2, c2c = col_divs[1]

    # Extract sub-grid between dividers (inclusive)
    sub = [list(grid[r][c1:c2 + 1]) for r in range(r1, r2 + 1)]
    sh, sw = len(sub), len(sub[0])

    # Find pattern color in interior
    pattern = None
    for r in range(1, sh - 1):
        for c in range(1, sw - 1):
            if sub[r][c] != bg:
                pattern = sub[r][c]
                break
        if pattern is not None:
            break

    if pattern is None:
        return sub

    # Fill from each pattern pixel toward the matching border
    if pattern == c2c:  # right border -> fill rightward
        for r in range(1, sh - 1):
            leftmost = -1
            for c in range(1, sw - 1):
                if sub[r][c] == pattern:
                    leftmost = c
                    break
            if leftmost != -1:
                for c in range(leftmost, sw - 1):
                    sub[r][c] = pattern
    elif pattern == c1c:  # left border -> fill leftward
        for r in range(1, sh - 1):
            rightmost = -1
            for c in range(sw - 2, 0, -1):
                if sub[r][c] == pattern:
                    rightmost = c
                    break
            if rightmost != -1:
                for c in range(1, rightmost + 1):
                    sub[r][c] = pattern
    elif pattern == r2c:  # bottom border -> fill downward
        for c in range(1, sw - 1):
            topmost = -1
            for r in range(1, sh - 1):
                if sub[r][c] == pattern:
                    topmost = r
                    break
            if topmost != -1:
                for r in range(topmost, sh - 1):
                    sub[r][c] = pattern
    elif pattern == r1c:  # top border -> fill upward
        for c in range(1, sw - 1):
            bottommost = -1
            for r in range(sh - 2, 0, -1):
                if sub[r][c] == pattern:
                    bottommost = r
                    break
            if bottommost != -1:
                for r in range(1, bottommost + 1):
                    sub[r][c] = pattern

    return sub
