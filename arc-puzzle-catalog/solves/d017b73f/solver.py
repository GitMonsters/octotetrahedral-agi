def solve(grid: list[list[int]]) -> list[list[int]]:
    """Remove separator columns between objects and vertically shift each object
    to form a connected zigzag chain. Gravity direction is determined by whether
    the first object touches the top or bottom row."""
    rows = len(grid)
    cols = len(grid[0])

    # Extract objects separated by all-zero columns
    objects = []
    obj_cols: list[int] = []
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            if obj_cols:
                objects.append([[grid[r][cc] for cc in obj_cols] for r in range(rows)])
                obj_cols = []
        else:
            obj_cols.append(c)
    if obj_cols:
        objects.append([[grid[r][cc] for cc in obj_cols] for r in range(rows)])

    if len(objects) <= 1:
        if objects:
            return objects[0]
        return grid

    # Gravity: first object touches top row → UP, else DOWN
    first = objects[0]
    gravity_up = any(first[0][c] != 0 for c in range(len(first[0])))

    shifts = [0]
    prev_conn_row = None

    for i in range(1, len(objects)):
        prev_obj = objects[i - 1]
        curr_obj = objects[i]
        ps = shifts[i - 1]

        # Previous object's rightmost column (shifted) non-zero rows
        prev_right_nz: set[int] = set()
        for r in range(rows):
            src = r - ps
            if 0 <= src < rows and prev_obj[src][-1] != 0:
                prev_right_nz.add(r)

        # Current object's leftmost column original non-zero rows
        curr_left_nz = [r for r in range(rows) if curr_obj[r][0] != 0]

        # Bounding rows of current object
        nz = [r for r in range(rows) for c in range(len(curr_obj[0])) if curr_obj[r][c] != 0]
        min_r, max_r = min(nz), max(nz)

        # Enumerate valid shifts
        candidates = []
        for s in range(-min_r, rows - max_r):
            shifted = set(r + s for r in curr_left_nz if 0 <= r + s < rows)
            shared = prev_right_nz & shifted
            if shared:
                candidates.append((s, shared))

        # Pick best (shift, connection_row)
        best = None
        for s, shared in candidates:
            for conn in shared:
                if prev_conn_row is None:
                    target = 0 if gravity_up else rows - 1
                    score = (1000 - abs(conn - target),
                             -s if gravity_up else s)
                else:
                    is_diff = int(conn != prev_conn_row)
                    dist = abs(conn - prev_conn_row)
                    grav = -s if gravity_up else s
                    score = (is_diff, dist, grav)

                if best is None or score > best[0]:
                    best = (score, s, conn)

        shifts.append(best[1] if best else 0)
        prev_conn_row = best[2] if best else prev_conn_row

    # Compose output
    total_w = sum(len(o[0]) for o in objects)
    out = [[0] * total_w for _ in range(rows)]
    col_off = 0
    for i, obj in enumerate(objects):
        s = shifts[i]
        w = len(obj[0])
        for r in range(rows):
            src = r - s
            if 0 <= src < rows:
                for c in range(w):
                    out[r][col_off + c] = obj[src][c]
        col_off += w
    return out
