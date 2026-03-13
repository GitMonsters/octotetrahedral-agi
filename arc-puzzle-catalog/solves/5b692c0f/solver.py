def solve(grid):
    """Each shape has a line of 4s as symmetry axis. Mirror the fuller side onto the sparser side."""
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find connected components of non-zero cells
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                component = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if 0 <= cr < rows and 0 <= cc < cols and not visited[cr][cc] and grid[cr][cc] != 0:
                        visited[cr][cc] = True
                        component.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((cr + dr, cc + dc))
                components.append(component)

    for comp in components:
        fours = [(r, c) for r, c in comp if grid[r][c] == 4]
        if not fours:
            continue

        # Find the longest continuous line of 4s (the symmetry axis)
        best_axis = None
        best_len = 0

        by_row = {}
        for r, c in fours:
            by_row.setdefault(r, []).append(c)
        for r, cs in by_row.items():
            cs.sort()
            i = 0
            while i < len(cs):
                j = i
                while j + 1 < len(cs) and cs[j + 1] == cs[j] + 1:
                    j += 1
                ln = j - i + 1
                if ln > best_len:
                    best_len = ln
                    best_axis = ('h', r, list(range(cs[i], cs[j] + 1)))
                i = j + 1

        by_col = {}
        for r, c in fours:
            by_col.setdefault(c, []).append(r)
        for c, rs in by_col.items():
            rs.sort()
            i = 0
            while i < len(rs):
                j = i
                while j + 1 < len(rs) and rs[j + 1] == rs[j] + 1:
                    j += 1
                ln = j - i + 1
                if ln > best_len:
                    best_len = ln
                    best_axis = ('v', c, list(range(rs[i], rs[j] + 1)))
                i = j + 1

        if best_axis is None or best_len < 2:
            continue

        # Clear all component cells from result
        for r, c in comp:
            result[r][c] = 0

        if best_axis[0] == 'h':
            axis_row = best_axis[1]
            axis_cols = best_axis[2]
            for c in axis_cols:
                result[axis_row][c] = 4

            above = {(r, c): grid[r][c] for r, c in comp if r < axis_row}
            below = {(r, c): grid[r][c] for r, c in comp if r > axis_row}

            if len(above) >= len(below):
                template = above
                max_dist = max(axis_row - r for r, _ in template) if template else 0
                for d in range(1, max_dist + 1):
                    src_row = axis_row - d
                    dst_row = axis_row + d
                    if 0 <= src_row < rows and 0 <= dst_row < rows:
                        for c in range(cols):
                            if (src_row, c) in template:
                                val = template[(src_row, c)]
                                result[src_row][c] = val
                                result[dst_row][c] = val
            else:
                template = below
                max_dist = max(r - axis_row for r, _ in template) if template else 0
                for d in range(1, max_dist + 1):
                    src_row = axis_row + d
                    dst_row = axis_row - d
                    if 0 <= src_row < rows and 0 <= dst_row < rows:
                        for c in range(cols):
                            if (src_row, c) in template:
                                val = template[(src_row, c)]
                                result[src_row][c] = val
                                result[dst_row][c] = val

        else:  # vertical axis
            axis_col = best_axis[1]
            axis_rows_list = best_axis[2]
            for r in axis_rows_list:
                result[r][axis_col] = 4

            left = {(r, c): grid[r][c] for r, c in comp if c < axis_col}
            right = {(r, c): grid[r][c] for r, c in comp if c > axis_col}

            if len(left) >= len(right):
                template = left
                max_dist = max(axis_col - c for _, c in template) if template else 0
                for d in range(1, max_dist + 1):
                    src_col = axis_col - d
                    dst_col = axis_col + d
                    if 0 <= src_col < cols and 0 <= dst_col < cols:
                        for r in range(rows):
                            if (r, src_col) in template:
                                val = template[(r, src_col)]
                                result[r][src_col] = val
                                result[r][dst_col] = val
            else:
                template = right
                max_dist = max(c - axis_col for _, c in template) if template else 0
                for d in range(1, max_dist + 1):
                    src_col = axis_col + d
                    dst_col = axis_col - d
                    if 0 <= src_col < cols and 0 <= dst_col < cols:
                        for r in range(rows):
                            if (r, src_col) in template:
                                val = template[(r, src_col)]
                                result[r][src_col] = val
                                result[r][dst_col] = val

    return result
