from collections import Counter


def _sign(x):
    return (x > 0) - (x < 0)


def _orbit(r, c, h, w):
    return {
        (r, c),
        (r, w - 1 - c),
        (h - 1 - r, c),
        (h - 1 - r, w - 1 - c),
    }


def _get(grid, r, c):
    h, w = len(grid), len(grid[0])
    if 0 <= r < h and 0 <= c < w:
        return grid[r][c]
    return None


def _detect_motifs(grid, bg):
    h, w = len(grid), len(grid[0])
    cr, cc = h // 2, w // 2
    motifs = {}

    for r in range(h):
        for c in range(w):
            corner = grid[r][c]
            if corner == bg:
                continue

            orth = []
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                value = _get(grid, r + dr, c + dc)
                if value is not None and value != bg:
                    orth.append((dr, dc, value))

            if len(orth) != 2:
                continue

            a, b = orth
            if (a[0] == 0) == (b[0] == 0):
                continue
            if a[2] != b[2] or a[2] == corner:
                continue

            vr = a if a[0] else b
            hz = a if a[1] else b
            sr = _sign(vr[0])
            sc = _sign(hz[1])
            edge = a[2]
            diag_r, diag_c = r + 2 * sr, c + 2 * sc
            diag = _get(grid, diag_r, diag_c)

            if diag == edge:
                kind = "full"
            elif diag == corner:
                kind = "checker"
            else:
                kind = "L"

            toward = sr == _sign(cr - r) and sc == _sign(cc - c)
            corners = tuple(sorted(_orbit(r, c, h, w)))
            key = (corner, edge, corners)

            cells = {(r, c), (r + 2 * sr, c), (r, c + 2 * sc)}
            if diag in (corner, edge):
                cells.add((diag_r, diag_c))

            candidate = {
                "corner": corner,
                "edge": edge,
                "kind": kind,
                "toward": toward,
                "corners": corners,
                "cells": cells,
            }

            prev = motifs.get(key)
            if prev is None or ((not prev["toward"]) and toward):
                motifs[key] = candidate

    return list(motifs.values())


def _draw_direct(out, top, bottom, left, right, color, full):
    for c in range(left + 2, right, 2):
        out[top][c] = color
        out[bottom][c] = color
    for r in range(top + 2, bottom, 2):
        out[r][left] = color
        out[r][right] = color
    if full and top + 2 <= bottom - 2 and left + 2 <= right - 2:
        for rr in (top + 2, bottom - 2):
            for cc in (left + 2, right - 2):
                out[rr][cc] = color


def _draw_outer(out, top, bottom, left, right, color):
    h, w = len(out), len(out[0])
    tr, br = top - 2, bottom + 2
    lc, rc = left - 2, right + 2
    if 0 <= tr < h and 0 <= br < h:
        for c in range(left, right + 1, 2):
            out[tr][c] = color
            out[br][c] = color
    if 0 <= lc < w and 0 <= rc < w:
        for r in range(top, bottom + 1, 2):
            out[r][lc] = color
            out[r][rc] = color


def transform(grid):
    bg = Counter(cell for row in grid for cell in row).most_common(1)[0][0]
    h, w = len(grid), len(grid[0])
    out = [[bg] * w for _ in range(h)]

    motifs = _detect_motifs(grid, bg)
    used = set()
    for motif in motifs:
        used.update(motif["cells"])

    for motif in motifs:
        top = min(r for r, _ in motif["corners"])
        bottom = max(r for r, _ in motif["corners"])
        left = min(c for _, c in motif["corners"])
        right = max(c for _, c in motif["corners"])

        if motif["kind"] == "L" and not motif["toward"]:
            _draw_outer(out, top, bottom, left, right, motif["edge"])
        else:
            _draw_direct(
                out,
                top,
                bottom,
                left,
                right,
                motif["edge"],
                motif["kind"] == "full",
            )

    for motif in motifs:
        for r, c in motif["corners"]:
            out[r][c] = motif["corner"]

    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == bg or (r, c) in used:
                continue
            for rr, cc in _orbit(r, c, h, w):
                out[rr][cc] = color

    return out
