from __future__ import annotations

from typing import List, Tuple, Optional, Set


Grid = List[List[int]]


DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
    "NE": (-1, 1),
    "NW": (-1, -1),
    "SE": (1, 1),
    "SW": (1, -1),
}


def _find_separator_color(grid: Grid) -> Optional[int]:
    """Infer the color used for the full-row/full-col separators."""
    h, w = len(grid), len(grid[0])

    row_vals = []
    for r in range(h):
        if all(v == grid[r][0] for v in grid[r]):
            row_vals.append(grid[r][0])

    col_vals = []
    for c in range(w):
        if all(grid[r][c] == grid[0][c] for r in range(h)):
            col_vals.append(grid[0][c])

    if not row_vals and not col_vals:
        return None

    # Prefer a value that appears in BOTH full rows and full cols.
    inter = set(row_vals) & set(col_vals)
    candidates = list(inter) if inter else (row_vals + col_vals)

    # Most frequent candidate wins.
    best = None
    best_count = -1
    for v in set(candidates):
        cnt = candidates.count(v)
        if cnt > best_count:
            best = v
            best_count = cnt
    return best


def _separator_indices(grid: Grid, sep: int) -> Tuple[List[int], List[int]]:
    h, w = len(grid), len(grid[0])
    sep_rows = [r for r in range(h) if all(v == sep for v in grid[r])]
    sep_cols = [c for c in range(w) if all(grid[r][c] == sep for r in range(h))]
    return sep_rows, sep_cols


def _spans(n: int, cuts: List[int]) -> List[Tuple[int, int]]:
    cuts2 = [-1] + sorted(cuts) + [n]
    spans: List[Tuple[int, int]] = []
    for a, b in zip(cuts2, cuts2[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _dir_for_one(
    ri: int,
    ci: int,
    R: int,
    C: int,
    cell_hw: List[List[Tuple[int, int]]],
    has1: List[List[bool]],
) -> Optional[str]:
    """Direction rule learned from training examples."""
    h, w = cell_hw[ri][ci]
    top, bottom, left, right = ri == 0, ri == R - 1, ci == 0, ci == C - 1

    if top and left:
        return "E"
    if top and right:
        return "S"
    if bottom and left:
        return None
    if bottom and right:
        return "W"

    # Right-edge lasers only when stacked under a 1-block.
    if right:
        return "S" if (ri > 0 and has1[ri - 1][ci]) else None

    if bottom:
        return "NE"

    if not (top or bottom or left or right):
        # Interior: tall vs wide determines diagonal.
        return "NW" if h > w else "SW"

    return None


def _predict_two_blocks(
    has1: List[List[bool]],
    has9: List[List[bool]],
    cell_hw: List[List[Tuple[int, int]]],
) -> Set[Tuple[int, int]]:
    R, C = len(has1), len(has1[0])
    twos: Set[Tuple[int, int]] = set()

    for ri in range(R):
        for ci in range(C):
            if not has1[ri][ci]:
                continue

            d = _dir_for_one(ri, ci, R, C, cell_hw, has1)
            if not d:
                continue

            dr, dc = DIRS[d]
            r, c = ri + dr, ci + dc

            if dr != 0 and dc != 0:
                # Diagonal: mark ONLY the first 9 encountered.
                while 0 <= r < R and 0 <= c < C and (not has1[r][c]) and (not has9[r][c]):
                    r += dr
                    c += dc
                if 0 <= r < R and 0 <= c < C and has9[r][c]:
                    twos.add((r, c))
            else:
                # Orthogonal: skip empties, then mark a consecutive run of 9-blocks.
                while 0 <= r < R and 0 <= c < C and (not has1[r][c]) and (not has9[r][c]):
                    r += dr
                    c += dc
                while 0 <= r < R and 0 <= c < C and has9[r][c]:
                    twos.add((r, c))
                    r += dr
                    c += dc

    return twos


def transform(grid: Grid) -> Grid:
    h, w = len(grid), len(grid[0])

    sep_color = _find_separator_color(grid)
    if sep_color is None:
        # Fallback: no separators detected; nothing to do.
        return [row[:] for row in grid]

    sep_rows, sep_cols = _separator_indices(grid, sep_color)
    row_spans = _spans(h, sep_rows)
    col_spans = _spans(w, sep_cols)

    # If we couldn't form a block grid, return unchanged.
    if not row_spans or not col_spans:
        return [row[:] for row in grid]

    R, C = len(row_spans), len(col_spans)

    has1 = [[False] * C for _ in range(R)]
    has9 = [[False] * C for _ in range(R)]
    cell_hw: List[List[Tuple[int, int]]] = [[(0, 0)] * C for _ in range(R)]

    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            cell_h = r1 - r0 + 1
            cell_w = c1 - c0 + 1
            cell_hw[ri][ci] = (cell_h, cell_w)

            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    v = grid[r][c]
                    if v == 1:
                        has1[ri][ci] = True
                    elif v == 9:
                        has9[ri][ci] = True

    two_blocks = _predict_two_blocks(has1, has9, cell_hw)

    out = [row[:] for row in grid]

    # Fill each cell block uniformly.
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            if has9[ri][ci]:
                fill = 2 if (ri, ci) in two_blocks else 8
            else:
                fill = 7

            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    out[r][c] = fill

    return out
