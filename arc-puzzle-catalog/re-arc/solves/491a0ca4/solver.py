def find_largest_solid_rectangle(grid, color):
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    best_area = 0
    best_rect = None
    for r1 in range(n_rows):
        for c1 in range(n_cols):
            if grid[r1][c1] != color:
                continue
            for r2 in range(r1, n_rows):
                for c2 in range(c1, n_cols):
                    is_solid = True
                    for r in range(r1, r2 + 1):
                        for c in range(c1, c2 + 1):
                            if grid[r][c] != color:
                                is_solid = False
                                break
                        if not is_solid:
                            break
                    if is_solid:
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area > best_area:
                            best_area = area
                            best_rect = (r1, c1, r2, c2)
    return best_rect


def transform(grid):
    n_rows = len(grid)
    rect = find_largest_solid_rectangle(grid, 7)
    if rect is None:
        return []
    r1, c1, r2, c2 = rect
    mirror_r1 = n_rows - 1 - r1
    mirror_r2 = n_rows - 1 - r2
    output = []
    for r in range(mirror_r1, mirror_r2 - 1, -1):
        row = [grid[r][c] for c in range(c1, c2 + 1)]
        output.append(row)
    return output
