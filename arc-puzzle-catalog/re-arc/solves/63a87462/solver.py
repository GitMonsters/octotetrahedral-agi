from collections import Counter
from itertools import product


def transform(grid):
    bg = _find_bg(grid)
    groups = _get_non_bg(grid, bg)
    if not groups:
        return [row[:] for row in grid]

    frame_color, frame_info = None, None
    for color, cells in groups.items():
        rect = _is_rectangle_corners(cells)
        if rect:
            frame_color, frame_info = color, rect

    if frame_info:
        _, _, H, W = frame_info
        return _solve_with_dims(grid, bg, groups, frame_color, H, W)

    heights, widths = [], []
    for cells in groups.values():
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        heights.append(max(rows) - min(rows) + 1)
        widths.append(max(cols) - min(cols) + 1)
    for H in range(max(heights), max(heights) + 15):
        for W in range(max(widths), max(widths) + 15):
            result = _solve_with_dims(grid, bg, groups, None, H, W)
            if result is not None:
                return result
    return None


def _find_bg(grid):
    return Counter(v for row in grid for v in row).most_common(1)[0][0]


def _get_non_bg(grid, bg):
    cells = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != bg:
                cells.setdefault(grid[r][c], []).append((r, c))
    return cells


def _is_rectangle_corners(cells):
    if len(cells) != 4:
        return None
    rows = sorted(set(r for r, c in cells))
    cols = sorted(set(c for r, c in cells))
    if len(rows) == 2 and len(cols) == 2:
        if set(cells) == set(product(rows, cols)):
            return (rows[0], cols[0], rows[1] - rows[0] + 1, cols[1] - cols[0] + 1)
    return None


def _on_border(r, c, H, W):
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def _is_corner(r, c, H, W):
    return (r in (0, H - 1)) and (c in (0, W - 1))


def _map_cells_valid(cells, r0, c0, H, W, is_frame=False):
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    if (min_r - r0) % H + (max_r - min_r) >= H:
        return None
    if (min_c - c0) % W + (max_c - min_c) >= W:
        return None
    mapped, seen = [], set()
    for r, c in cells:
        mr, mc = (r - r0) % H, (c - c0) % W
        if not _on_border(mr, mc, H, W):
            return None
        if not is_frame and _is_corner(mr, mc, H, W):
            return None
        if (mr, mc) in seen:
            return None
        seen.add((mr, mc))
        mapped.append((mr, mc))
    return mapped


def _can_be_border_placed(cells, H, W):
    for r0 in range(H):
        for c0 in range(W):
            if _map_cells_valid(cells, r0, c0, H, W) is not None:
                return True
    return False


def _find_all_valid(cells, H, W, occupied, is_frame=False):
    results = []
    for r0 in range(H):
        for c0 in range(W):
            mapped = _map_cells_valid(cells, r0, c0, H, W, is_frame)
            if mapped is None:
                continue
            if any((mr, mc) in occupied for mr, mc in mapped):
                continue
            results.append((r0, c0, mapped))
    return results


def _split_group(cells, H, W):
    if len(cells) <= 1:
        return [cells]
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    if max(rows) - min(rows) < H and max(cols) - min(cols) < W:
        if _can_be_border_placed(cells, H, W):
            return [cells]
    gaps = []
    for dim, vals in [('row', rows), ('col', cols)]:
        uv = sorted(set(vals))
        for i in range(len(uv) - 1):
            gaps.append((uv[i + 1] - uv[i], dim, uv[i]))
    gaps.sort(reverse=True)
    for _, dim, threshold in gaps:
        if dim == 'col':
            c1 = [(r, c) for r, c in cells if c <= threshold]
            c2 = [(r, c) for r, c in cells if c > threshold]
        else:
            c1 = [(r, c) for r, c in cells if r <= threshold]
            c2 = [(r, c) for r, c in cells if r > threshold]
        if c1 and c2:
            result = []
            for cluster in [c1, c2]:
                result.extend(_split_group(cluster, H, W))
            return result
    return [[(r, c)] for r, c in cells]


def _solve_with_dims(grid, bg, groups, frame_color, H, W):
    units = []
    if frame_color is not None:
        units.append((frame_color, groups[frame_color], True))
    for color in sorted(groups.keys()):
        if color == frame_color:
            continue
        for cluster in _split_group(groups[color], H, W):
            units.append((color, cluster, False))

    def backtrack(idx, occupied):
        if idx == len(units):
            return True, {}
        color, cells, is_frame = units[idx]
        valid = _find_all_valid(cells, H, W, occupied, is_frame)
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        dr, dc = min_r % H, min_c % W
        valid.sort(key=lambda e: (
            -(100 * (e[0] == dr) + 100 * (e[1] == dc)),
            e[0], e[1]
        ))
        for r0, c0, mapped in valid:
            new_occ = dict(occupied)
            for mr, mc in mapped:
                new_occ[(mr, mc)] = color
            ok, asn = backtrack(idx + 1, new_occ)
            if ok:
                asn[idx] = (r0, c0)
                return True, asn
        return False, {}

    ok, asn = backtrack(0, {})
    if not ok:
        return None
    out = [[bg] * W for _ in range(H)]
    for i, (color, cells, _) in enumerate(units):
        r0, c0 = asn[i]
        for r, c in cells:
            out[(r - r0) % H][(c - c0) % W] = color
    return out
