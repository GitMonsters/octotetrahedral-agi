"""Solver for 42992bc2: Grid with colored shapes and rectangular frame outlines.
Shapes matching a marker pattern inside the frame are changed to the marker color.
When no marker exists, shapes with mismatched hole counts or small anomalous fragments are removed."""

from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    cnt = Counter(c for row in grid for c in row)
    bg = cnt.most_common(1)[0][0]

    def get_comps(color):
        visited = [[False]*W for _ in range(H)]
        comps = []
        for r in range(H):
            for c in range(W):
                if grid[r][c] == color and not visited[r][c]:
                    stack = [(r, c)]; visited[r][c] = True; cells = []
                    while stack:
                        cr, cc = stack.pop(); cells.append((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True; stack.append((nr, nc))
                    comps.append(cells)
        return comps

    def is_rect_outline(cells):
        cs = set(cells)
        if len(cs) < 8: return False
        rmin = min(r for r, c in cs); rmax = max(r for r, c in cs)
        cmin = min(c for r, c in cs); cmax = max(c for r, c in cs)
        if rmax - rmin < 2 or cmax - cmin < 2: return False
        expected = set()
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if r in (rmin, rmax) or c in (cmin, cmax):
                    expected.add((r, c))
        return cs == expected

    def normalize(cells):
        rmin = min(r for r, c in cells); cmin = min(c for r, c in cells)
        return frozenset((r - rmin, c - cmin) for r, c in cells)

    def bbox_dims(cells):
        rmin = min(r for r, c in cells); rmax = max(r for r, c in cells)
        cmin = min(c for r, c in cells); cmax = max(c for r, c in cells)
        return (rmax - rmin + 1, cmax - cmin + 1)

    all_colors = set(c for row in grid for c in row) - {bg}
    frames = []
    frame_color = None
    for color in all_colors:
        for comp in get_comps(color):
            if is_rect_outline(comp):
                frames.append(comp)
                frame_color = color

    if not frames:
        return out

    fc = set(frames[0])
    r1, c1 = min(r for r, c in fc), min(c for r, c in fc)
    r2, c2 = max(r for r, c in fc), max(c for r, c in fc)
    interior_cells = [(r, c) for r in range(r1 + 1, r2) for c in range(c1 + 1, c2)]

    marker_color = None
    for cr, cc in interior_cells:
        v = grid[cr][cc]
        if v != bg and v != frame_color:
            marker_color = v
            break

    marker_pattern = None
    if marker_color is not None:
        marker_cells = [(r, c) for r, c in interior_cells if grid[r][c] == marker_color]
        marker_pattern = normalize(marker_cells)

    remaining = all_colors - {frame_color}
    if marker_color is not None:
        remaining -= {marker_color}
    shape_color = max(remaining, key=lambda c: cnt[c]) if remaining else frame_color

    frame_cell_set = set()
    for f in frames:
        frame_cell_set.update(f)
    shape_comps = [comp for comp in get_comps(shape_color)
                   if not any((r, c) in frame_cell_set for r, c in comp)]

    if marker_pattern is not None:
        for comp in shape_comps:
            if normalize(comp) == marker_pattern:
                for r, c in comp:
                    out[r][c] = marker_color
    else:
        shape_info = []
        for comp in shape_comps:
            h, w = bbox_dims(comp)
            holes = h * w - len(comp)
            shape_info.append((comp, holes, len(comp)))

        holed = [(comp, holes, sz) for comp, holes, sz in shape_info if holes > 0]
        majority_holes = 0
        if holed:
            hcnt = Counter(h for _, h, _ in holed)
            majority_holes = hcnt.most_common(1)[0][0]

        for comp, holes, sz in shape_info:
            if sz <= 1:
                continue
            if holes > 0:
                if holes != majority_holes:
                    for r, c in comp:
                        out[r][c] = 0
            else:
                if majority_holes > 0 and sz <= majority_holes:
                    for r, c in comp:
                        out[r][c] = 0

    return out
