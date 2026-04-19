from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    flat = [c for r in grid for c in r]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = set(flat) - {bg}

    # Find marker color (all cells isolated - no same-color orthogonal neighbors)
    marker = None
    for color in sorted(non_bg):
        cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        ok = True
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == color:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            marker = color
            break

    shape_colors = non_bg - {marker} if marker is not None else non_bg

    # Fill markers: adjacent to shape -> shape color, else -> bg
    filled = [row[:] for row in grid]
    marker_adj: dict[tuple[int, int], bool] = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == marker:
                adj_shape = None
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in shape_colors:
                        adj_shape = grid[nr][nc]
                        break
                filled[r][c] = adj_shape if adj_shape is not None else bg
                marker_adj[(r, c)] = (adj_shape is not None)

    # Find column groups from ORIGINAL non-bg columns
    orig_non_bg_cols = sorted(set(
        c for r in range(rows) for c in range(cols) if grid[r][c] != bg
    ))
    if not orig_non_bg_cols:
        return [[bg] * cols for _ in range(rows)]

    groups: list[list[int]] = []
    cur = [orig_non_bg_cols[0]]
    for c in orig_non_bg_cols[1:]:
        if c == cur[-1] + 1:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)

    # Check if each group has shape content
    def has_shape(g: list[int]) -> bool:
        for c in g:
            for r in range(rows):
                if grid[r][c] in shape_colors:
                    return True
                if grid[r][c] == marker and marker_adj.get((r, c), False):
                    return True
        return False

    group_has_shape = [has_shape(g) for g in groups]

    # Classify each gap between consecutive groups
    gap_info: list[tuple] = []
    for i in range(len(groups) - 1):
        right_col = groups[i][-1]
        left_col = groups[i + 1][0]

        right_markers = [(r, c) for (r, c) in marker_adj if c == right_col]
        left_markers = [(r, c) for (r, c) in marker_adj if c == left_col]

        right_adj = any(marker_adj[rc] for rc in right_markers) if right_markers else False
        left_adj = any(marker_adj[rc] for rc in left_markers) if left_markers else False

        if right_adj and left_adj:
            rm_row = right_markers[0][0]
            lm_row = left_markers[0][0]
            gap_info.append(('collapse', rm_row, lm_row))
        elif right_adj or left_adj:
            gap_info.append(('preserve',))
        else:
            gap_info.append(('remove',))

    # Determine which groups to include
    included = [False] * len(groups)
    for i in range(len(groups)):
        if group_has_shape[i]:
            included[i] = True
        else:
            if i > 0 and gap_info[i - 1][0] == 'preserve':
                included[i] = True
            if i < len(groups) - 1 and gap_info[i][0] == 'preserve':
                included[i] = True

    # Build output columns: (original_col, vertical_offset) pairs
    out_cols: list[tuple[int, int]] = []
    current_offset = 0
    prev_included = -1

    for i in range(len(groups)):
        if not included[i]:
            continue

        if prev_included >= 0:
            gap_idx = i - 1
            gi = gap_info[gap_idx]

            if gi[0] == 'collapse':
                current_offset = current_offset + (gi[1] - gi[2])
            elif gi[0] == 'preserve':
                current_offset = 0
                gap_start = groups[prev_included][-1] + 1
                gap_end = groups[i][0]
                for c in range(gap_start, gap_end):
                    out_cols.append((c, 0))
        else:
            current_offset = 0

        for c in groups[i]:
            out_cols.append((c, current_offset))

        prev_included = i

    # Assemble output grid
    out_width = len(out_cols)
    out = [[bg] * out_width for _ in range(rows)]

    for out_c, (src_c, v_off) in enumerate(out_cols):
        for src_r in range(rows):
            dst_r = src_r + v_off
            if 0 <= dst_r < rows:
                out[dst_r][out_c] = filled[src_r][src_c]

    return out
