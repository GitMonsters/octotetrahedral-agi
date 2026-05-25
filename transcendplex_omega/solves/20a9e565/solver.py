"""Solver for 20a9e565 — Staircase tile continuation"""
import json
from typing import List
from collections import Counter


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])

    # Find white cells → output bbox
    white_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    wr = [r for r, c in white_cells]
    wc = [c for r, c in white_cells]
    out_r0, out_r1 = min(wr), max(wr)
    out_c0, out_c1 = min(wc), max(wc)
    out_h = out_r1 - out_r0 + 1
    out_w = out_c1 - out_c0 + 1

    # Find pattern cells (non-zero, non-white)
    pattern_cells = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols)
                     if grid[r][c] not in (0, 5)]

    # Check if every pattern row uses a single color
    row_colors: dict[int, set] = {}
    for r, c, v in pattern_cells:
        row_colors.setdefault(r, set()).add(v)
    all_single_color = all(len(cs) == 1 for cs in row_colors.values())

    # Group pattern columns into contiguous groups
    all_cols = sorted(set(c for r, c, v in pattern_cells))
    col_groups: list[list[int]] = []
    for c in all_cols:
        if not col_groups or c > col_groups[-1][-1] + 1:
            col_groups.append([c])
        else:
            col_groups[-1].append(c)

    # Route based on pattern structure
    if len(col_groups) == 1 and all_single_color:
        return _solve_horizontal_bands(pattern_cells, row_colors,
                                       out_r0, out_c0, out_h, out_w)
    if len(col_groups) == 1:
        return _solve_nested_frames(pattern_cells,
                                    out_r0, out_c0, out_h, out_w)
    return _solve_staircase(pattern_cells, col_groups,
                            out_r0, out_c0, out_h, out_w)


# ── horizontal zigzag bands (e.g. test 0) ──────────────────────────

def _solve_horizontal_bands(pattern_cells, row_colors,
                            out_r0, out_c0, out_h, out_w):
    # Group consecutive rows of the same color into bands
    bands: list[dict] = []
    cur = None
    for r in sorted(row_colors):
        color = next(iter(row_colors[r]))
        if cur is None or r > cur['rows'][-1] + 1 or color != cur['color']:
            cur = {'rows': [r], 'color': color}
            bands.append(cur)
        else:
            cur['rows'].append(r)

    band_height = len(bands[0]['rows'])
    y_period = (bands[1]['rows'][0] - bands[0]['rows'][0]
                if len(bands) > 1 else band_height + 1)

    # Filled columns per row in the widest band
    widest = max(bands, key=lambda b:
        max(c for r, c, v in pattern_cells if r in b['rows'])
        - min(c for r, c, v in pattern_cells if r in b['rows']))
    row_filled = []
    for r in widest['rows']:
        row_filled.append(sorted(set(c for r2, c, v in pattern_cells if r2 == r)))

    # Find smallest x-period consistent with the band pattern
    x_period = None
    for p in range(2, 100):
        ok = True
        for cols in row_filled:
            if not cols:
                continue
            mods = set(c % p for c in cols)
            fset = set(cols)
            if not all(((c % p) in mods) == (c in fset)
                       for c in range(min(cols), max(cols) + 1)):
                ok = False
                break
        if ok:
            x_period = p
            break

    # Motif: filled offsets within x-period for each row in the band, plus gap
    motif = [set(c % x_period for c in cols) if cols else set()
             for cols in row_filled]
    motif.extend(set() for _ in range(y_period - band_height))

    # Color cycle across bands
    bcolors = [b['color'] for b in bands]
    for cl in range(1, len(bcolors) + 1):
        if all(bcolors[i] == bcolors[i % cl] for i in range(len(bcolors))):
            break
    cycle = bcolors[:cl]
    first = bands[0]['rows'][0]

    output = [[0] * out_w for _ in range(out_h)]
    for r in range(out_h):
        off = (out_r0 + r) - first
        bi = off // y_period
        ri = off % y_period
        if ri < len(motif) and motif[ri]:
            color = cycle[bi % cl]
            for c in range(out_w):
                if ((out_c0 + c) % x_period) in motif[ri]:
                    output[r][c] = color
    return output


# ── nested concentric frames (e.g. test 1) ─────────────────────────

def _solve_nested_frames(pattern_cells, out_r0, out_c0, out_h, out_w):
    rows_with = sorted(set(r for r, c, v in pattern_cells))

    # Split into steps at gap rows
    steps: list[list[int]] = [[rows_with[0]]]
    for i in range(1, len(rows_with)):
        if rows_with[i] > rows_with[i - 1] + 1:
            steps.append([rows_with[i]])
        else:
            steps[-1].append(rows_with[i])

    # Center strip from level-0 of the last (widest) step
    last = steps[-1]
    l0_top = last[0]
    l0_cols = sorted(c for r, c, v in pattern_cells if r == l0_top)
    center_width = max(l0_cols) - min(l0_cols) + 1
    center_col = (min(l0_cols) + max(l0_cols)) / 2.0

    # Bar width from level 1 top row (left contiguous run)
    bar_width = 3
    if len(last) >= 4:
        l1_cols = sorted(c for r, c, v in pattern_cells if r == last[2])
        bar = [l1_cols[0]]
        for c in l1_cols[1:]:
            if c == bar[-1] + 1:
                bar.append(c)
            else:
                break
        bar_width = len(bar)

    # Two pattern colors and level-0 color per step
    colors = sorted(set(v for r, c, v in pattern_cells))
    cl0 = min(l0_cols)
    step_c0s = [next(v for r, c, v in pattern_cells if r == s[0] and c == cl0)
                for s in steps]
    # Output level-0 color continues the alternation
    level0_color = [c for c in colors if c != step_c0s[-1]][0]

    n_blocks = out_h // 2
    output = [[0] * out_w for _ in range(out_h)]

    for lv in range(n_blocks):
        w = center_width + 4 * lv
        left = round(center_col - w / 2.0 + 0.5) - out_c0
        oc = level0_color if lv % 2 == 0 else [c for c in colors if c != level0_color][0]
        ic = [c for c in colors if c != oc][0]
        tr, br = lv * 2, lv * 2 + 1

        if lv == 0:
            for dc in range(w):
                if 0 <= left + dc < out_w:
                    output[tr][left + dc] = oc
            if br < out_h:
                for pos in (left, left + w - 1):
                    if 0 <= pos < out_w:
                        output[br][pos] = oc
        else:
            for dc in range(bar_width):
                for pos in (left + dc, left + w - 1 - dc):
                    if 0 <= pos < out_w:
                        output[tr][pos] = oc
            if br < out_h:
                for pos in (left, left + w - 1):
                    if 0 <= pos < out_w:
                        output[br][pos] = oc
                for pos in (left + bar_width - 1, left + w - bar_width):
                    if 0 <= pos < out_w:
                        output[br][pos] = ic
    return output


# ── staircase tiles (trains 0-2) ───────────────────────────────────

def _solve_staircase(pattern_cells, col_groups,
                     out_r0, out_c0, out_h, out_w):
    tiles = []
    for cg in col_groups:
        c_min, c_max = min(cg), max(cg)
        tile_w = c_max - c_min + 1
        tile_rows = sorted(set(r for r, c, v in pattern_cells if c_min <= c <= c_max))
        r_min, r_max = min(tile_rows), max(tile_rows)
        color_count = Counter(v for r, c, v in pattern_cells if c_min <= c <= c_max)
        color = color_count.most_common(1)[0][0]
        pat = {}
        for r, c, v in pattern_cells:
            if c_min <= c <= c_max:
                pat[(r - r_min, c - c_min)] = v
        tiles.append({
            'col_min': c_min, 'col_max': c_max, 'row_min': r_min, 'row_max': r_max,
            'width': tile_w, 'height': r_max - r_min + 1, 'color': color, 'pattern': pat
        })
    tiles.sort(key=lambda t: t['col_min'])

    widths = [t['width'] for t in tiles]
    colors = [t['color'] for t in tiles]
    all_same_width = len(set(widths)) == 1

    for cycle_len in range(1, len(colors) + 1):
        if all(colors[i] == colors[i % cycle_len] for i in range(len(colors))):
            break
    color_cycle = colors[:cycle_len]

    if all_same_width and widths[0] == 2:
        # WIDTH-2 CONSTANT: staircase with color transitions
        rightmost = max(tiles, key=lambda t: t['col_min'])
        rightmost_idx = tiles.index(rightmost)

        step_c = tiles[1]['col_min'] - tiles[0]['col_min'] if len(tiles) >= 2 else 3
        extra_steps = round((out_c0 - rightmost['col_min']) / abs(step_c))
        if extra_steps == 0:
            extra_steps = 1

        own_color = color_cycle[(rightmost_idx + extra_steps) % cycle_len]
        total_pairs = len(tiles) + extra_steps
        start_side = 'left' if total_pairs % 2 == 1 else 'right'

        n_full = (out_h + 1) // 2
        n_trans = out_h // 2
        start_idx = color_cycle.index(own_color)
        full_colors = [color_cycle[(start_idx + i) % cycle_len] for i in range(n_full)]

        output = []
        side = start_side
        for i in range(n_full):
            c = full_colors[i]
            output.append([c, c])
            if i < n_trans:
                next_c = full_colors[i + 1] if i + 1 < n_full else 0
                if side == 'left':
                    output.append([next_c, c])
                else:
                    output.append([c, next_c])
                side = 'right' if side == 'left' else 'left'
        return output[:out_h]

    else:
        # GROWING WIDTH or CONSTANT WIDTH > 2
        tw = tiles[0]['width']
        first_tile = tiles[0]
        th = first_tile['height']
        partial_row = None
        for dr in range(th):
            row_cells = [(dc, v) for (r, dc), v in first_tile['pattern'].items() if r == dr]
            if len(row_cells) < tw:
                partial_row = [0] * tw
                for dc, v in row_cells:
                    partial_row[dc] = 1
                break
        if partial_row is None:
            partial_row = [1] * tw

        dw = widths[1] - widths[0] if len(widths) > 1 else 0
        if dw != 0:
            out_tile_idx = round((out_w - widths[0]) / dw)
        else:
            gap = tiles[1]['col_min'] - tiles[0]['col_max'] - 1 if len(tiles) >= 2 else 1
            out_tile_idx = len(tiles)
            col_pos = tiles[-1]['col_max'] + 1 + gap
            while col_pos < out_c0:
                out_tile_idx += 1
                col_pos += tw + gap

        out_color = color_cycle[out_tile_idx % cycle_len]

        if dw != 0 and out_w != tw:
            scaled_partial = [0] * out_w
            for i in range(tw):
                if partial_row[i]:
                    if i == 0:
                        scaled_partial[0] = 1
                    elif i == tw - 1:
                        scaled_partial[out_w - 1] = 1
        else:
            scaled_partial = partial_row[:out_w] + [0] * max(0, out_w - len(partial_row))

        output = [[0] * out_w for _ in range(out_h)]
        for r in range(out_h):
            if r % 2 == 0:
                output[r] = [out_color] * out_w
            else:
                output[r] = [out_color if scaled_partial[c] else 0 for c in range(out_w)]
        return output


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            res = solve(ex['input'])
            ok = res == ex['output']
            print(f"{split.title()} {i}: {'PASS ✓' if ok else 'FAIL ✗'}")
