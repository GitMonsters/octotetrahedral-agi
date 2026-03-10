"""898e7135 — Template holes filled by colored blocks at N× scale.

The input contains:
1. A large colored rectangular template with black (0) holes inside
2. Several colored blocks (each made of N×N sub-blocks) on the black background
3. Scattered single-cell noise markers (ignored)

The output scales the template by N× (each cell → N×N block) and fills each
hole region with the color of the matching block. N is auto-detected (2× or 3×).
Matching is done by comparing the hole's shape with each block's reduced shape
under all 8 rotations/reflections.
"""
from collections import deque


def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])

    # Find connected components of same-color non-black cells
    visited: set[tuple[int, int]] = set()
    components: list[tuple[int, set[tuple[int, int]]]] = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and (r, c) not in visited:
                color = grid[r][c]
                comp: set[tuple[int, int]] = set()
                queue = deque([(r, c)])
                while queue:
                    cr, cc = queue.popleft()
                    if (cr, cc) in visited or grid[cr][cc] != color:
                        continue
                    visited.add((cr, cc))
                    comp.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and grid[nr][nc] == color:
                            queue.append((nr, nc))
                components.append((color, comp))

    # Template = largest connected component
    components.sort(key=lambda x: len(x[1]), reverse=True)
    tmpl_color, tmpl_cells = components[0]

    # Template bounding box
    tr0 = min(r for r, c in tmpl_cells)
    tc0 = min(c for r, c in tmpl_cells)
    tr1 = max(r for r, c in tmpl_cells)
    tc1 = max(c for r, c in tmpl_cells)
    tmpl_h = tr1 - tr0 + 1
    tmpl_w = tc1 - tc0 + 1

    # Extract template grid (local coords)
    tmpl = [[grid[tr0 + r][tc0 + c] for c in range(tmpl_w)] for r in range(tmpl_h)]

    # Find connected hole regions (black cells inside template bbox)
    hole_vis: set[tuple[int, int]] = set()
    holes: list[set[tuple[int, int]]] = []
    for r in range(tmpl_h):
        for c in range(tmpl_w):
            if tmpl[r][c] == 0 and (r, c) not in hole_vis:
                region: set[tuple[int, int]] = set()
                queue = deque([(r, c)])
                while queue:
                    cr, cc = queue.popleft()
                    if (cr, cc) in hole_vis:
                        continue
                    hole_vis.add((cr, cc))
                    region.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < tmpl_h and 0 <= nc < tmpl_w and (nr, nc) not in hole_vis and tmpl[nr][nc] == 0:
                            queue.append((nr, nc))
                holes.append(region)

    # Find colored blocks (non-template components, area >= 4 to skip noise)
    raw_blocks: list[tuple[int, set[tuple[int, int]]]] = []
    for bcolor, bcomp in components[1:]:
        if len(bcomp) < 4:
            continue
        raw_blocks.append((bcolor, bcomp))

    # Detect scale factor (blocks may be 2×, 3×, etc.)
    scale = _detect_scale(raw_blocks, holes)

    blocks: list[tuple[int, frozenset[tuple[int, int]]]] = []
    for bcolor, bcomp in raw_blocks:
        br0 = min(r for r, c in bcomp)
        bc0 = min(c for r, c in bcomp)
        reduced = set()
        for r, c in bcomp:
            reduced.add(((r - br0) // scale, (c - bc0) // scale))
        norm = _normalize(reduced)
        blocks.append((bcolor, norm))

    # Match blocks to holes by shape (with rotation/reflection)
    hole_color: dict[int, int] = {}
    used: set[int] = set()
    for hi, region in enumerate(holes):
        nreg = _normalize(region)
        for bi, (bc, bs) in enumerate(blocks):
            if bi in used:
                continue
            if len(bs) != len(nreg):
                continue
            if nreg in _all_orientations(bs):
                hole_color[hi] = bc
                used.add(bi)
                break

    # Build scaled output
    oh, ow = tmpl_h * scale, tmpl_w * scale
    out = [[tmpl_color] * ow for _ in range(oh)]
    for hi, region in enumerate(holes):
        c = hole_color.get(hi, 0)
        for r, col in region:
            for dr in range(scale):
                for dc in range(scale):
                    out[r * scale + dr][col * scale + dc] = c
    return out


def _normalize(s):
    r0 = min(r for r, c in s)
    c0 = min(c for r, c in s)
    return frozenset((r - r0, c - c0) for r, c in s)


def _detect_scale(raw_blocks, holes):
    """Try scale factors 2..9 and return the first where all blocks match a hole."""
    hole_orients: set[frozenset[tuple[int, int]]] = set()
    for h in holes:
        hole_orients.update(_all_orientations(_normalize(h)))

    for scale in range(2, 10):
        matched = 0
        for _bcolor, bcomp in raw_blocks:
            br0 = min(r for r, c in bcomp)
            bc0 = min(c for r, c in bcomp)
            reduced: set[tuple[int, int]] = set()
            for r, c in bcomp:
                reduced.add(((r - br0) // scale, (c - bc0) // scale))
            if _normalize(reduced) in hole_orients:
                matched += 1
        if matched == len(raw_blocks):
            return scale
    return 2


def _all_orientations(s):
    pts = list(s)
    results = set()
    for _ in range(4):
        results.add(_normalize(frozenset(pts)))
        pts = [(c, -r) for r, c in pts]
    pts = [(r, -c) for r, c in list(s)]
    for _ in range(4):
        results.add(_normalize(frozenset(pts)))
        pts = [(c, -r) for r, c in pts]
    return results
