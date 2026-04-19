from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

Grid = List[List[int]]
Pt = Tuple[int, int]


def transform(input_grid: Grid) -> Grid:
    """Solve ARC-AGI puzzle 186b1380.

    The large input contains several monochrome motifs (often corner pieces /
    partial frames) on a dominant background. For each non-background color we:
      1) pick the most plausible local cluster of components for that color,
      2) embed it into an odd-sized square, choose an offset that maximizes
         border coverage after mirroring,
      3) mirror 4-way (horizontal/vertical) to form a symmetric square pattern,
      4) center and nest all patterns into the final output (size = largest N).

    Training examples also show a special case where large (k>=4) L-corner groups
    are filled into right triangles before mirroring.
    """

    bg = _background_color(input_grid)

    patterns: List[Tuple[int, int, Set[Pt]]] = []  # (N, color, mask_pts)
    colors = sorted({v for row in input_grid for v in row if v != bg})

    for col in colors:
        comps = _connected_components(input_grid, col)
        if not comps:
            continue

        # Candidates: use all components for the color, or any group with identical
        # (bbox_h, bbox_w, cell_count). This helps ignore distant distractors.
        candidates: List[Tuple[List[Pt], Dict[str, object]]] = []
        candidates.append(([p for comp in comps for p in comp["cells"]], {"type": "all"}))

        groups: Dict[Tuple[int, int, int], List[Dict[str, object]]] = defaultdict(list)
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            groups[(r1 - r0 + 1, c1 - c0 + 1, comp["size"])].append(comp)

        for key, clist in groups.items():
            cells = [p for comp in clist for p in comp["cells"]]
            candidates.append((cells, {"type": "group", "key": key, "count": len(clist)}))

        best: Tuple[Tuple[int, int, int], int, Set[Pt]] | None = None
        for cells, meta in candidates:
            if not cells:
                continue

            r0, c0, r1, c1 = _union_bbox(cells)
            bh, bw = (r1 - r0 + 1), (c1 - c0 + 1)
            N = max(bh, bw)
            if N % 2 == 0:
                continue

            rel = [(r - r0, c - c0) for r, c in cells]
            sym = _best_offset_sym(rel, N, bh, bw)

            # Special case: large L-corner groups (k>=4) become filled triangles.
            fill_k = None
            if meta.get("type") == "group":
                gbh, gbw, gsz = meta["key"]  # type: ignore[misc]
                count = int(meta["count"])  # type: ignore[misc]
                if gbh == gbw:
                    k = int(gbh)
                    if count >= 4 and int(gsz) == 2 * k - 1 and k >= 4:
                        fill_k = k

            if fill_k is not None:
                sym = _mirror_4way(_triangle_corner(fill_k), N)

            score = (len(cells), N, len(sym))
            if best is None or score > best[0]:
                best = (score, N, sym)

        if best is not None:
            _, N, sym = best
            patterns.append((N, col, sym))

    if not patterns:
        return [row[:] for row in input_grid]

    M = max(N for N, _, _ in patterns)
    out: Grid = [[bg for _ in range(M)] for _ in range(M)]

    # Nest: larger patterns first.
    for N, col, mask in sorted(patterns, key=lambda x: -x[0]):
        off = (M - N) // 2
        for r, c in mask:
            out[off + r][off + c] = col

    return out


def _background_color(grid: Grid) -> int:
    cnt: Counter[int] = Counter()
    for row in grid:
        cnt.update(row)
    return cnt.most_common(1)[0][0]


def _connected_components(grid: Grid, color: int) -> List[Dict[str, object]]:
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    comps: List[Dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] != color:
                continue

            q: deque[Pt] = deque([(r, c)])
            seen[r][c] = True
            cells: List[Pt] = []

            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx][ny] and grid[nx][ny] == color:
                        seen[nx][ny] = True
                        q.append((nx, ny))

            r0, c0, r1, c1 = _union_bbox(cells)
            comps.append({"cells": cells, "bbox": (r0, c0, r1, c1), "size": len(cells)})

    return comps


def _union_bbox(cells: Iterable[Pt]) -> Tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _mirror_4way(points: Set[Pt], N: int) -> Set[Pt]:
    sym: Set[Pt] = set()
    for r, c in points:
        for rr in (r, N - 1 - r):
            for cc in (c, N - 1 - c):
                sym.add((rr, cc))
    return sym


def _best_offset_sym(rel_cells: List[Pt], N: int, bh: int, bw: int) -> Set[Pt]:
    """Place a (bh x bw) patch inside NxN, mirror 4-way, choose best offset."""

    best: Tuple[Tuple[int, int, int], Set[Pt]] | None = None
    for ro in range(N - bh + 1):
        for co in range(N - bw + 1):
            pts = {(ro + r, co + c) for r, c in rel_cells}
            sym = _mirror_4way(pts, N)
            border = sum(1 for r, c in sym if r in (0, N - 1) or c in (0, N - 1))
            score = (border, len(sym), -(ro + co))
            if best is None or score > best[0]:
                best = (score, sym)

    assert best is not None
    return best[1]


def _triangle_corner(k: int) -> Set[Pt]:
    """Filled right triangle in the top-left kxk corner."""

    return {(r, c) for r in range(k) for c in range(k - r)}
