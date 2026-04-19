def transform(grid):
    from collections import Counter
    H, W = len(grid), len(grid[0])
    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]

    color_positions = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != bg:
                if v not in color_positions:
                    color_positions[v] = []
                color_positions[v].append((r, c))

    colors = list(color_positions.keys())
    if len(colors) != 2:
        return grid

    cA, cB = colors
    posA, posB = color_positions[cA], color_positions[cB]

    def detect_edge(positions):
        if all(r == 0 for r, c in positions): return 'top'
        if all(r == H - 1 for r, c in positions): return 'bottom'
        if all(c == 0 for r, c in positions): return 'left'
        if all(c == W - 1 for r, c in positions): return 'right'
        return None

    edgeA = detect_edge(posA)
    edgeB = detect_edge(posB)

    if len(posA) <= len(posB):
        fill_color, fill_edge, fill_pos = cA, edgeA, posA
        step_edge, step_pos = edgeB, posB
    else:
        fill_color, fill_edge, fill_pos = cB, edgeB, posB
        step_edge, step_pos = edgeA, posA

    if step_edge in ('top', 'bottom'):
        step_coords = sorted(set(c for _, c in step_pos))
    else:
        step_coords = sorted(set(r for r, _ in step_pos))

    if fill_edge in ('top', 'bottom'):
        fill_starts = sorted(set(c for _, c in fill_pos))
    else:
        fill_starts = sorted(set(r for r, _ in fill_pos))

    out = [row[:] for row in grid]

    if step_edge in ('top', 'bottom'):
        if fill_edge == 'left':
            seg_bounds = [0] + step_coords + [W]
            segments = [(seg_bounds[i], seg_bounds[i + 1] - 1) for i in range(len(seg_bounds) - 1)]
            shift_dir = -1 if step_edge == 'bottom' else 1
            for start_row in fill_starts:
                cur_row = start_row
                for s, e in segments:
                    if s > e:
                        continue
                    for c in range(s, e + 1):
                        if 0 <= cur_row < H:
                            out[cur_row][c] = fill_color
                    cur_row += shift_dir

        elif fill_edge == 'right':
            seg_bounds = [W - 1] + sorted(step_coords, reverse=True) + [-1]
            segments = []
            for i in range(len(seg_bounds) - 1):
                hi = seg_bounds[i]
                lo = seg_bounds[i + 1] + 1
                if lo <= hi:
                    segments.append((lo, hi))
            shift_dir = -1 if step_edge == 'bottom' else 1
            for start_row in fill_starts:
                cur_row = start_row
                for lo, hi in segments:
                    for c in range(lo, hi + 1):
                        if 0 <= cur_row < H:
                            out[cur_row][c] = fill_color
                    cur_row += shift_dir
    else:
        if fill_edge == 'bottom':
            seg_bounds = [H - 1] + sorted(step_coords, reverse=True) + [-1]
            segments = []
            for i in range(len(seg_bounds) - 1):
                hi = seg_bounds[i]
                lo = seg_bounds[i + 1] + 1
                if lo <= hi:
                    segments.append((lo, hi))
            shift_dir = -1 if step_edge == 'right' else 1
            for start_col in fill_starts:
                cur_col = start_col
                for lo, hi in segments:
                    for r in range(lo, hi + 1):
                        if 0 <= r < H and 0 <= cur_col < W:
                            out[r][cur_col] = fill_color
                    cur_col += shift_dir

        elif fill_edge == 'top':
            seg_bounds = [0] + step_coords + [H]
            segments = [(seg_bounds[i], seg_bounds[i + 1] - 1) for i in range(len(seg_bounds) - 1)]
            shift_dir = -1 if step_edge == 'right' else 1
            for start_col in fill_starts:
                cur_col = start_col
                for s, e in segments:
                    if s > e:
                        continue
                    for r in range(s, e + 1):
                        if 0 <= r < H and 0 <= cur_col < W:
                            out[r][cur_col] = fill_color
                    cur_col += shift_dir

    return out
