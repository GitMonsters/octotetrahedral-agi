"""Solver for ARC puzzle b6f77b65.

The grid contains nested L-shaped bracket borders forming a staircase or grid
pattern. The indicator color at (0,0) identifies which bars to remove. When
removed, the remaining structure collapses downward.
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    indicator = grid[0][0]

    has_ind = any(
        grid[r][c] == indicator
        for r in range(H) for c in range(W)
        if (r, c) != (0, 0)
    )
    if not has_ind:
        return [row[:] for row in grid]

    ind_cells = {
        (r, c) for r in range(H) for c in range(W)
        if grid[r][c] == indicator and (r, c) != (0, 0)
    }
    ind_rows = {r for r, c in ind_cells}
    min_ind_row = min(ind_rows)

    # Check if shift needed: any non-indicator cell at topmost indicator row
    # is unsupported below (empty or indicator cell directly below)
    needs_shift = False
    for c in range(W):
        v = grid[min_ind_row][c]
        if v != 0 and v != indicator and (min_ind_row, c) != (0, 0):
            below_r = min_ind_row + 1
            if below_r >= H:
                continue
            bv = grid[below_r][c]
            if bv == 0 or bv == indicator:
                needs_shift = True
                break

    if not needs_shift:
        out = [row[:] for row in grid]
        for r, c in ind_cells:
            out[r][c] = 0
        return out

    # Find removed vertical and horizontal bars
    removed_v: list[tuple[int, int, int]] = []
    for c in range(W):
        r = 0
        while r < H:
            if (r, c) in ind_cells:
                s = r
                while r < H and (r, c) in ind_cells:
                    r += 1
                if r - s >= 2:
                    removed_v.append((s, r - 1, c))
            else:
                r += 1

    removed_h: list[tuple[int, int, int]] = []
    for rr in range(H):
        c = 0
        while c < W:
            if (rr, c) in ind_cells:
                s = c
                while c < W and (rr, c) in ind_cells:
                    c += 1
                if c - s >= 2:
                    removed_h.append((rr, s, c - 1))
            else:
                c += 1

    v_bar_cols = {rc for _, _, rc in removed_v}

    # Find non-indicator horizontal bars
    non_ind_h_bars: list[tuple[int, int, int, int, int]] = []
    for rr in range(H):
        c = 0
        while c < W:
            v = grid[rr][c]
            if v != 0 and v != indicator and (rr, c) != (0, 0):
                s = c
                while c < W and grid[rr][c] == v:
                    c += 1
                if c - s >= 2:
                    non_ind_h_bars.append((rr, s, c - 1, v, c - s))
            else:
                c += 1
    if not non_ind_h_bars:
        out = [row[:] for row in grid]
        for r, c in ind_cells:
            out[r][c] = 0
        return out

    outer_bar = max(non_ind_h_bars, key=lambda b: b[4])
    outer_top_row = outer_bar[0]
    all_rows = [
        r for r in range(1, H)
        if any(grid[r][c] != 0 and (r, c) != (0, 0) for c in range(W))
    ]
    max_row = max(all_rows)

    # Compute base_shift: if indicator has horizontal bar below outer bar,
    # use it for shift computation
    ind_h_below = [rr for rr, _, _ in removed_h if rr > outer_top_row]
    has_ind_h_below = len(ind_h_below) > 0
    if has_ind_h_below:
        lowest_ind_h = min(ind_h_below)
        base_shift = max_row - lowest_ind_h + 1
    else:
        base_shift = max_row - outer_top_row

    min_nz_row = min(all_rows)
    inner_corner_col = None
    for c in range(W):
        if (grid[min_nz_row][c] != 0 and grid[min_nz_row][c] != indicator
                and (min_nz_row, c) != (0, 0)):
            v = grid[min_nz_row][c]
            if min_nz_row + 1 < H and grid[min_nz_row + 1][c] == v:
                inner_corner_col = c
                break

    inner_bracket_end = min_nz_row
    if inner_corner_col is not None:
        corner_color = grid[min_nz_row][inner_corner_col]
        r = min_nz_row
        while r + 1 < H and grid[r + 1][inner_corner_col] == corner_color:
            r += 1
        inner_bracket_end = r

    inner_cols: set[int] = set()
    for r in range(min_nz_row, inner_bracket_end + 1):
        for c in range(W):
            if grid[r][c] != 0 and grid[r][c] != indicator and (r, c) != (0, 0):
                inner_cols.add(c)
    inner_max_c = max(inner_cols) if inner_cols else 0

    inter_extra, inter_min_row = 0, H
    for r1, r2, rc in removed_v:
        if r2 < outer_top_row:
            inter_extra = max(inter_extra, r2 - r1)
            inter_min_row = min(inter_min_row, r1)
    for rr, c1, c2 in removed_h:
        if rr < outer_top_row:
            inter_extra = max(inter_extra, 1)
            inter_min_row = min(inter_min_row, rr)

    nz_outer = {
        c for c in range(W)
        for r in range(outer_top_row, max_row + 1)
        if grid[r][c] != 0 and (r, c) != (0, 0)
    }
    if not nz_outer:
        out = [row[:] for row in grid]
        for r, c in ind_cells:
            out[r][c] = 0
        return out
    min_outer_c, max_outer_c = min(nz_outer), max(nz_outer)
    left_removed = any(rc == min_outer_c for _, _, rc in removed_v)
    right_removed = any(rc == max_outer_c for _, _, rc in removed_v)

    center_col = None
    if inner_corner_col is not None:
        c = inner_corner_col
        vals = [
            grid[r][c]
            for r in range(outer_top_row, max_row + 1)
            if grid[r][c] != 0 and grid[r][c] != indicator
        ]
        if len(vals) >= 2 and len(set(vals)) == 1:
            center_col = c
    two_wings = center_col is not None

    inner_far_edge_col = None
    if two_wings and left_removed:
        inner_far_edge_col = inner_max_c

    kept_bar_cells: set[tuple[int, int]] = set()
    for c in [min_outer_c, max_outer_c]:
        v = grid[outer_top_row][c]
        if v != 0 and v != indicator:
            for r in range(outer_top_row, max_row + 1):
                if grid[r][c] == v:
                    kept_bar_cells.add((r, c))

    bottom_edge_cols = set()
    if has_ind_h_below:
        bottom_edge_cols = {min_outer_c, max_outer_c}

    out = [[0] * W for _ in range(H)]
    for c in range(W):
        out[0][c] = grid[0][c]

    stays_cells: list[tuple[int, int, int]] = []
    shift_cells: list[tuple[int, int, int]] = []

    for c in range(W):
        col_cells = [
            (r, grid[r][c]) for r in range(1, H)
            if grid[r][c] != 0 and grid[r][c] != indicator
        ]
        if not col_cells:
            continue

        segs: list[list[tuple[int, int]]] = []
        cur = [col_cells[0]]
        for i in range(1, len(col_cells)):
            if col_cells[i][0] == cur[-1][0] + 1:
                cur.append(col_cells[i])
            else:
                segs.append(cur[:])
                cur = [col_cells[i]]
        segs.append(cur[:])

        on_kept_side = False
        if two_wings:
            if right_removed and c < center_col:
                on_kept_side = True
            elif left_removed and c > center_col:
                on_kept_side = True

        for seg in segs:
            seg_min, seg_max = seg[0][0], seg[-1][0]
            seg_in_kept = any((r, c) in kept_bar_cells for r, _ in seg)
            grounded = seg_max == max_row
            below_inner = seg_min > inner_bracket_end
            at_far_edge = (
                inner_far_edge_col is not None
                and c == inner_far_edge_col
                and seg_min <= inner_bracket_end
            )

            if has_ind_h_below and grounded and c not in bottom_edge_cols:
                grounded = False

            stays = (
                seg_in_kept
                or (on_kept_side and below_inner)
                or grounded
                or (on_kept_side and at_far_edge)
            )

            if stays:
                for r, v in seg:
                    stays_cells.append((r, c, v))
            else:
                shift = base_shift
                if not has_ind_h_below and c in v_bar_cols:
                    if any(
                        r1 > seg_max and rc == c
                        for r1, _, rc in removed_v
                    ):
                        shift = max(shift, (H - 1) - seg_max)
                if inter_extra > 0:
                    has_above = seg_min < inter_min_row
                    near_v = any(
                        abs(c - rc) <= 1 and r1 <= seg_min <= r2
                        for r1, r2, rc in removed_v if r2 < outer_top_row
                    )
                    if has_above and c != inner_corner_col:
                        shift += inter_extra
                    elif not has_above and near_v and c != inner_corner_col:
                        shift += inter_extra
                for r, v in seg:
                    nr = r + shift
                    if 0 <= nr < H:
                        shift_cells.append((nr, c, v))

    # Write stays first, then shifted (shifted overwrites)
    for r, c, v in stays_cells:
        out[r][c] = v
    for r, c, v in shift_cells:
        out[r][c] = v

    # Extend vertical bars through removed horizontal bar gaps
    for rr, c1, c2 in removed_h:
        if rr >= outer_top_row:
            continue
        for c in range(c1, c2 + 1):
            if rr > 0 and grid[rr - 1][c] != 0 and grid[rr - 1][c] != indicator:
                above_v = grid[rr - 1][c]
                if rr > 1 and grid[rr - 2][c] == above_v:
                    ext_r = rr + base_shift
                    if 0 <= ext_r < H and out[ext_r][c] == 0:
                        out[ext_r][c] = above_v

    return out


if __name__ == "__main__":
    import os
    task_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "dataset", "tasks", "b6f77b65.json"
    )
    with open(task_path) as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        ok = result == expected
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    for i, ex in enumerate(data["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            ok = result == ex["output"]
            print(f"Test {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
        else:
            print(f"Test {i}: produced {len(result)}x{len(result[0])} output")

    if all_pass:
        print("\nALL PASS")
