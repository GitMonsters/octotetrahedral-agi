def transform(grid):
    bg = _background(grid)
    comps = _components(grid, bg)
    items = []
    for i, cells in enumerate(comps):
        crop, color, r0, c0 = _crop(grid, cells, bg)
        items.append(
            {
                "id": i,
                "cells": cells,
                "crop": crop,
                "color": color,
                "r0": r0,
                "c0": c0,
                "h": len(crop),
                "w": len(crop[0]),
                "kind": _classify(crop, color),
            }
        )

    top_hbar = _top_hbar(items)
    lshape = _find_lshape(items, grid, bg, exclude_id=top_hbar["id"] if top_hbar else None)
    complex_block = _first(
        items,
        lambda x: x["kind"] == "complex_block" and (not lshape or x["id"] not in lshape["member_ids"]),
    )
    rightish = _find_rightish(items, grid, bg, exclude_id=top_hbar["id"] if top_hbar else None)

    if top_hbar and complex_block and rightish:
        return _solve_family2(bg, top_hbar, complex_block, rightish)
    if top_hbar and lshape:
        used = set(lshape["member_ids"]) | {top_hbar["id"]}
        extra = _first(items, lambda x: x["id"] not in used)
        return _solve_family1(bg, top_hbar, lshape, extra)
    return [row[:] for row in grid]


def _solve_family1(bg, hbar, lshape, extra):
    width = hbar["w"] + 2
    height = lshape["h"] + 1
    out = [[bg for _ in range(width)] for _ in range(height)]

    for c in range(hbar["w"]):
        out[0][1 + c] = hbar["color"]

    mid = 1 + (hbar["w"] - 1) // 2
    out[1][mid] = hbar["color"]

    start_c = 0 if lshape["kind"] == "l_left" else width - lshape["w"]
    _overlay(out, lshape["crop"], 1, start_c, bg)
    out[height - 2][mid] = lshape["color"]

    if extra:
        spine_col = _spine_col(extra["crop"], extra["color"])
        leaf_col = _top_leaf_col(extra["crop"], extra["color"], spine_col)
        for r in range(2, max(2, height - 2)):
            if r < height - 1:
                out[r][mid] = extra["color"]
        if leaf_col is not None:
            delta = leaf_col - spine_col
            leaf_out_col = mid + delta
            if 0 <= leaf_out_col < width:
                out[1][leaf_out_col] = extra["color"]

    return out


def _solve_family2(bg, hbar, complex_block, rightish):
    width = hbar["w"] + 2
    height = rightish["h"] + 2
    out = [[bg for _ in range(width)] for _ in range(height)]

    for c in range(hbar["w"]):
        out[0][1 + c] = hbar["color"]

    start_c = width - rightish["w"]
    _overlay(out, rightish["crop"], 1, start_c, bg)

    right_col = width - 1
    vert_len = _right_col_count(rightish["crop"], rightish["color"])
    if vert_len:
        mid_row = 1 + (vert_len - 1) // 2
        out[mid_row][right_col - 1] = rightish["color"]

    skeleton = _block_skeleton(complex_block["crop"], complex_block["color"], bg)
    _overlay(out, skeleton, 1, 1, bg)
    return out


def _background(grid):
    counts = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def _components(grid, bg):
    h = len(grid)
    w = len(grid[0])
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == bg or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == color and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _crop(grid, cells, bg):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    color = grid[cells[0][0]][cells[0][1]]
    crop = [[bg for _ in range(c1 - c0 + 1)] for _ in range(r1 - r0 + 1)]
    for r, c in cells:
        crop[r - r0][c - c0] = color
    return crop, color, r0, c0


def _classify(crop, color):
    occ = [(r, c) for r in range(len(crop)) for c in range(len(crop[0])) if crop[r][c] == color]
    rows = {r for r, _ in occ}
    cols = {c for _, c in occ}
    if len(rows) == 1:
        return "hbar"
    if len(cols) == 1:
        return "vbar"
    if _has_full_2x2(crop, color):
        return "complex_block"
    return "other"


def _top_hbar(items):
    hbars = [x for x in items if x["kind"] == "hbar"]
    if not hbars:
        return None
    return min(hbars, key=lambda x: (x["r0"], x["c0"]))


def _find_lshape(items, grid, bg, exclude_id=None):
    best = None
    vbars = [x for x in items if x["kind"] == "vbar" and x["id"] != exclude_id]
    hbars = [x for x in items if x["kind"] == "hbar" and x["id"] != exclude_id]
    for v in vbars:
        for h in hbars:
            if v["id"] == h["id"] or v["color"] != h["color"]:
                continue
            if h["r0"] <= v["r0"]:
                continue
            kind = "l_left" if h["c0"] > v["c0"] else "l_right"
            crop, r0, c0 = _union_crop(grid, [v, h], bg)
            cand = {
                "color": v["color"],
                "crop": crop,
                "r0": r0,
                "c0": c0,
                "h": len(crop),
                "w": len(crop[0]),
                "kind": kind,
                "member_ids": {v["id"], h["id"]},
            }
            score = (r0, c0)
            if best is None or score < best[0]:
                best = (score, cand)
    return best[1] if best else None


def _find_rightish(items, grid, bg, exclude_id=None):
    vbar = _first(items, lambda x: x["kind"] == "vbar" and x["id"] != exclude_id)
    l_right = None
    lshape = _find_lshape(items, grid, bg, exclude_id=exclude_id)
    if lshape and lshape["kind"] == "l_right":
        l_right = lshape
    if l_right and vbar:
        if l_right["r0"] <= vbar["r0"]:
            return l_right
        return vbar
    return l_right or vbar


def _union_crop(grid, parts, bg):
    rows = []
    cols = []
    for part in parts:
        rows.extend([r for r, _ in part["cells"]])
        cols.extend([c for _, c in part["cells"]])
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    crop = [[bg for _ in range(c1 - c0 + 1)] for _ in range(r1 - r0 + 1)]
    color = parts[0]["color"]
    for part in parts:
        for r, c in part["cells"]:
            crop[r - r0][c - c0] = color
    return crop, r0, c0


def _has_full_2x2(crop, color):
    for r in range(len(crop) - 1):
        for c in range(len(crop[0]) - 1):
            if crop[r][c] == crop[r + 1][c] == crop[r][c + 1] == crop[r + 1][c + 1] == color:
                return True
    return False


def _block_skeleton(crop, color, bg):
    h = len(crop)
    w = len(crop[0])
    out = [[bg for _ in range(w)] for _ in range(h)]
    blocks = []
    for r in range(h - 1):
        for c in range(w - 1):
            if crop[r][c] == crop[r + 1][c] == crop[r][c + 1] == crop[r + 1][c + 1] == color:
                blocks.append((r, c))
    if blocks:
        r, c = max(blocks, key=lambda rc: (rc[1], -rc[0]))
        out[r][c] = color
        out[r + 1][c] = color
        out[r][c + 1] = color
        out[r + 1][c + 1] = color
    left_cols = [c for c in range(w) if any(crop[r][c] == color for r in range(h))]
    if left_cols:
        c = left_cols[0]
        rows = [r for r in range(h) if crop[r][c] == color]
        out[rows[0]][c] = color
        out[rows[-1]][c] = color
    return out


def _spine_col(crop, color):
    counts = [sum(1 for r in range(len(crop)) if crop[r][c] == color) for c in range(len(crop[0]))]
    best = max(counts)
    for c, count in enumerate(counts):
        if count == best:
            return c
    return 0


def _top_leaf_col(crop, color, spine_col):
    for c, value in enumerate(crop[0]):
        if value == color and c != spine_col:
            return c
    return None


def _right_col_count(crop, color):
    c = len(crop[0]) - 1
    return sum(1 for r in range(len(crop)) if crop[r][c] == color)


def _overlay(out, crop, r0, c0, bg):
    for r in range(len(crop)):
        for c in range(len(crop[0])):
            if crop[r][c] != bg:
                out[r0 + r][c0 + c] = crop[r][c]


def _first(items, pred):
    for item in items:
        if pred(item):
            return item
    return None
