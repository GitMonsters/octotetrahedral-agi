"""
Solver for ARC-AGI task 5545f144.

Rule: The grid has vertical separator columns dividing it into panels. Each panel
contains one multi-cell cluster (an 8-connected group) and scattered isolated cells.
The clusters across panels are rotations/reflections of the same base shape. Isolated
cells that appear in the same position in ALL panels are "common" anchors.

Output shape determination:
  - If one cluster orientation appears more than once across panels (majority),
    the output shape is the 180° rotation of that majority shape.
  - Otherwise (no majority or single panel), the output is the 90°-CCW rotation
    of the first panel's cluster.

Placement: The shape's "tip" cell (on the cluster's symmetry axis, in the most
isolated row/column) is mapped through the rotation and placed at the common
isolated cell that maximizes overlap with the first panel's pattern, breaking
ties by proximity to the original cluster.
"""

from collections import Counter
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    bg = _find_bg(grid)
    panels = _extract_panels(grid, bg)
    pr, pc = len(panels[0]), len(panels[0][0])
    panel_cells = [_get_cells(p, bg) for p in panels]

    p_iso, p_clust = [], []
    for pcs in panel_cells:
        comps = _cc8(pcs)
        p_clust.append(sorted([c for c in comps if len(c) > 1], key=len, reverse=True))
        p_iso.append({list(c)[0] for c in comps if len(c) == 1})

    common_iso = p_iso[0].copy()
    for iso in p_iso[1:]:
        common_iso &= iso

    cluster = p_clust[0][0]
    cluster_norm = _normalize(cluster)
    r0, c0 = next(iter(cluster))
    fg = panels[0][r0][c0]

    tip = _find_tip(cluster_norm)
    rows, cols = _get_bbox(cluster_norm)

    # Count normalized cluster shapes across panels
    base_isos = _all_isometries(cluster_norm)
    shape_counts: Counter = Counter()
    for clst_list in p_clust:
        for cl in clst_list:
            n = _normalize(cl)
            if n in base_isos:
                shape_counts[n] += 1

    majority = None
    for shape, count in shape_counts.most_common():
        if count > 1:
            majority = shape
            break

    output_shape, alignment_cell = _compute_output_and_alignment(
        cluster_norm, tip, majority, rows, cols
    )

    non_p0 = set()
    for pcs in panel_cells[1:]:
        non_p0 |= pcs

    candidates = []
    for anchor in common_iso:
        dr = anchor[0] - alignment_cell[0]
        dc = anchor[1] - alignment_cell[1]
        placed = frozenset((r + dr, c + dc) for r, c in output_shape)
        if any(r < 0 or r >= pr or c < 0 or c >= pc for r, c in placed):
            continue
        if len(placed & common_iso) != 1:
            continue
        if (placed - {anchor}) & non_p0:
            continue

        p0_ov = len(placed & panel_cells[0])
        cl_rows = (min(r for r, c in cluster), max(r for r, c in cluster))
        row_dist = max(0, cl_rows[0] - anchor[0]) + max(0, anchor[0] - cl_rows[1])
        cl_cols = (min(c for r, c in cluster), max(c for r, c in cluster))
        col_dist = max(0, cl_cols[0] - anchor[1]) + max(0, anchor[1] - cl_cols[1])

        candidates.append(
            {
                "placed": placed,
                "p0_ov": p0_ov,
                "row_dist": row_dist,
                "col_dist": col_dist,
            }
        )

    candidates.sort(key=lambda s: (-s["p0_ov"], s["row_dist"], s["col_dist"]))

    result = [[bg] * pc for _ in range(pr)]
    if candidates:
        for r, c in candidates[0]["placed"]:
            result[r][c] = fg
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_bg(grid):
    return Counter(v for row in grid for v in row).most_common(1)[0][0]


def _find_sep_cols(grid, bg):
    rows, cols = len(grid), len(grid[0])
    return [
        c
        for c in range(cols)
        if len(set(grid[r][c] for r in range(rows))) == 1 and grid[0][c] != bg
    ]


def _extract_panels(grid, bg):
    seps = _find_sep_cols(grid, bg)
    if not seps:
        return [grid]
    bounds = [-1] + seps + [len(grid[0])]
    return [
        [row[bounds[i] + 1 : bounds[i + 1]] for row in grid]
        for i in range(len(bounds) - 1)
        if bounds[i] + 1 < bounds[i + 1]
    ]


def _get_cells(panel, bg):
    return {
        (r, c)
        for r in range(len(panel))
        for c in range(len(panel[0]))
        if panel[r][c] != bg
    }


def _cc8(cells):
    remaining = set(cells)
    components = []
    while remaining:
        start = next(iter(remaining))
        comp = set()
        queue = [start]
        while queue:
            cell = queue.pop()
            if cell in comp:
                continue
            comp.add(cell)
            remaining.discard(cell)
            r, c = cell
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in remaining:
                        queue.append(nb)
        components.append(frozenset(comp))
    return components


def _normalize(cells):
    mr = min(r for r, c in cells)
    mc = min(c for r, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def _get_bbox(cells):
    return (
        max(r for r, c in cells) - min(r for r, c in cells) + 1,
        max(c for r, c in cells) - min(c for r, c in cells) + 1,
    )


def _all_isometries(cells):
    rows, cols = _get_bbox(cells)
    shapes = set()
    for fn in [
        lambda r, c: (r, c),
        lambda r, c: (c, rows - 1 - r),
        lambda r, c: (rows - 1 - r, cols - 1 - c),
        lambda r, c: (cols - 1 - c, r),
        lambda r, c: (r, cols - 1 - c),
        lambda r, c: (rows - 1 - r, c),
        lambda r, c: (c, r),
        lambda r, c: (cols - 1 - c, rows - 1 - r),
    ]:
        shapes.add(_normalize(frozenset(fn(r, c) for r, c in cells)))
    return shapes


def _find_tip(cells_norm):
    """Find the 'tip' cell on the shape's symmetry axis."""
    rows, cols = _get_bbox(cells_norm)
    is_h_sym = (
        _normalize(frozenset((r, cols - 1 - c) for r, c in cells_norm)) == cells_norm
    )
    is_v_sym = (
        _normalize(frozenset((rows - 1 - r, c) for r, c in cells_norm)) == cells_norm
    )

    def nbr_count(rc):
        return sum(
            1
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0) and (rc[0] + dr, rc[1] + dc) in cells_norm
        )

    if is_h_sym:
        axis_col = (cols - 1) / 2
        axis_cells = [(r, c) for r, c in cells_norm if abs(c - axis_col) < 0.01]
        row_counts = Counter(r for r, c in cells_norm)
        axis_cells.sort(key=lambda rc: (row_counts[rc[0]], nbr_count(rc)))
        return axis_cells[0]
    elif is_v_sym:
        axis_row = (rows - 1) / 2
        axis_cells = [(r, c) for r, c in cells_norm if abs(r - axis_row) < 0.01]
        col_counts = Counter(c for r, c in cells_norm)
        axis_cells.sort(key=lambda rc: (col_counts[rc[1]], nbr_count(rc)))
        return axis_cells[0]
    else:
        return min(cells_norm, key=nbr_count)


def _compute_output_and_alignment(cluster_norm, tip, majority, rows, cols):
    """Compute the output shape and the alignment cell (tip's image in the output)."""
    if majority is not None:
        maj_rows, maj_cols = _get_bbox(majority)
        output_shape = _normalize(
            frozenset((maj_rows - 1 - r, maj_cols - 1 - c) for r, c in majority)
        )

        # Find isometry mapping cluster_norm -> majority
        transforms = [
            lambda r, c: (r, c),
            lambda r, c: (c, rows - 1 - r),
            lambda r, c: (rows - 1 - r, cols - 1 - c),
            lambda r, c: (cols - 1 - c, r),
            lambda r, c: (r, cols - 1 - c),
            lambda r, c: (rows - 1 - r, c),
            lambda r, c: (c, r),
            lambda r, c: (cols - 1 - c, rows - 1 - r),
        ]
        p0_to_maj = None
        for fn in transforms:
            if _normalize(frozenset(fn(r, c) for r, c in cluster_norm)) == majority:
                p0_to_maj = fn
                break

        tip_raw = p0_to_maj(tip[0], tip[1])
        maj_cells_raw = frozenset(p0_to_maj(r, c) for r, c in cluster_norm)
        maj_mr = min(r for r, c in maj_cells_raw)
        maj_mc = min(c for r, c in maj_cells_raw)
        tip_in_maj = (tip_raw[0] - maj_mr, tip_raw[1] - maj_mc)

        tip_after = (maj_rows - 1 - tip_in_maj[0], maj_cols - 1 - tip_in_maj[1])
        out_raw = frozenset(
            (maj_rows - 1 - r, maj_cols - 1 - c) for r, c in majority
        )
        out_mr = min(r for r, c in out_raw)
        out_mc = min(c for r, c in out_raw)
        alignment = (tip_after[0] - out_mr, tip_after[1] - out_mc)
    else:
        output_shape = _normalize(
            frozenset((cols - 1 - c, r) for r, c in cluster_norm)
        )
        tip_raw = (cols - 1 - tip[1], tip[0])
        out_raw = frozenset((cols - 1 - c, r) for r, c in cluster_norm)
        out_mr = min(r for r, c in out_raw)
        out_mc = min(c for r, c in out_raw)
        alignment = (tip_raw[0] - out_mr, tip_raw[1] - out_mc)

    return output_shape, alignment


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, os

    task_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "dataset", "tasks", "5545f144.json",
    )
    with open(task_path) as f:
        data = json.load(f)

    all_pass = True
    for idx, pair in enumerate(data["train"] + data["test"]):
        is_test = idx >= len(data["train"])
        label = (
            f"test[{idx - len(data['train'])}]"
            if is_test
            else f"train[{idx}]"
        )
        result = solve(pair["input"])
        expected = pair["output"]
        ok = result == expected
        print(f"{label}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    raise SystemExit(0 if all_pass else 1)
