"""
DSL Program Synthesizer
========================
Given ARC train pairs, searches for a DSL program that explains all of them.

Three-layer strategy (fastest first):

  Layer 1 — Direct pattern recognizers  (~0.001–0.01 s each)
    Color mapping, geometric transform, invert+tile, region split+op,
    histogram bar chart, separator template stamp.

  Layer 2 — Object-based recognizers    (~0.01–0.5 s each)
    Shape-match recolor, frame fill, count-and-place, stamp-by-mapping
    (generalizes abc82100), concentric fill, run-length group.

  Layer 3 — Enumerative depth-1 sweep   (fallback, ~1–2 s)
    Try every single primitive over all color permutations.

Every strategy returns either a callable  g → Grid  or None.
The synthesizer tries them in order and stops at first hit.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from itertools import permutations
from typing import Callable, Dict, List, Optional, Set, Tuple

from dsl.primitives import (
    Grid, Object,
    rotate, flip_h, flip_v, flip_diag, flip_antidiag,
    crop, crop_bbox, crop_to_content,
    hstack, vstack,
    paint, stamp, clear_grid, empty_grid, copy_grid, fill_grid,
    recolor, apply_color_map, swap_colors,
    grid_or, grid_and, grid_xor,
    background, colors_in, color_histogram, colors_in,
    tile_grid, grid_size, grids_equal, output_size_matches,
    translate_object, object_to_grid, scale_object,
)
from dsl.perception import (
    find_objects, find_objects_of_color, find_objects_excluding,
    bounding_box, centroid, dist_sq, normalize_shape, object_orientations,
    object_color, object_colors, object_majority_color,
    find_separators, split_by_separators, find_separator_color,
    symmetry_flags,
    extract_color_chains, assign_objects_to_nearest,
)

# Callable type for a discovered program
Program = Callable[[Grid], Grid]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fits_all(prog: Program, pairs: List[Tuple[Grid, Grid]]) -> bool:
    """Return True if prog(x) == y for all (x, y) in pairs."""
    try:
        for x, y in pairs:
            if not grids_equal(prog(x), y):
                return False
        return True
    except Exception:
        return False

def _size_consistent(pairs: List[Tuple[Grid, Grid]]) -> bool:
    """True if all outputs have the same size."""
    if not pairs:
        return True
    h0, w0 = grid_size(pairs[0][1])
    return all(grid_size(y) == (h0, w0) for _, y in pairs)

def _output_same_size_as_input(pairs: List[Tuple[Grid, Grid]]) -> bool:
    return all(grid_size(x) == grid_size(y) for x, y in pairs)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Direct pattern recognizers
# ═══════════════════════════════════════════════════════════════════════════════

def try_identity(pairs):
    prog = lambda g: g
    return prog if _fits_all(prog, pairs) else None


def try_geometric_transforms(pairs):
    """Try all 8 rigid isometries: 4 rotations × 2 flips."""
    for fn in [
        flip_h, flip_v, flip_diag, flip_antidiag,
        lambda g: rotate(g, 1),
        lambda g: rotate(g, 2),
        lambda g: rotate(g, 3),
    ]:
        if _fits_all(fn, pairs):
            return fn
    # Combined: flip then rotate
    for flip in [flip_h, flip_v]:
        for k in range(1, 4):
            def make_fn(f=flip, r=k):
                return lambda g: rotate(f(g), r)
            fn = make_fn()
            if _fits_all(fn, pairs):
                return fn
    return None


def try_color_mapping(pairs):
    """Try bijective and non-bijective color remappings."""
    if not _output_same_size_as_input(pairs):
        return None

    # Collect color sets
    src_colors = sorted(colors_in(pairs[0][0]))
    dst_colors = sorted(colors_in(pairs[0][1]))

    # Build a mapping from training constraints
    # For each cell, the (input_color, output_color) pair must be consistent
    cmap: Dict[int, int] = {}
    consistent = True
    for x, y in pairs:
        h, w = grid_size(x)
        for r in range(h):
            for c in range(w):
                sc, dc = x[r][c], y[r][c]
                if sc in cmap:
                    if cmap[sc] != dc:
                        consistent = False
                        break
                else:
                    cmap[sc] = dc
            if not consistent:
                break
        if not consistent:
            break

    if consistent and cmap:
        prog = lambda g, m=cmap: apply_color_map(g, m)
        if _fits_all(prog, pairs):
            return prog
    return None


def try_invert_and_tile(pairs):
    """Invert 0↔color, then optionally tile 2×2 or other factor."""
    for x, y in pairs:
        if grid_size(x) == grid_size(y):
            return None  # output same size → not a tile task

    for x, y in pairs[:1]:
        hx, wx = grid_size(x)
        hy, wy = grid_size(y)
        if hy % hx != 0 or wy % wx != 0:
            continue
        tr, tc = hy // hx, wy // wx

        # Find non-zero color to invert against
        non_zeros = {c for row in x for c in row if c != 0}
        if len(non_zeros) != 1:
            continue
        color = next(iter(non_zeros))

        def make_fn(color=color, tr=tr, tc=tc):
            def fn(g):
                inv = [[color if v == 0 else 0 for v in row] for row in g]
                return tile_grid(inv, len(g)*tr, len(g[0])*tc)
            return fn

        prog = make_fn()
        if _fits_all(prog, pairs):
            return prog
    return None


def try_tiling(pairs):
    """Try simple tiling (no inversion): output = tile(input, r×c)."""
    for x, y in pairs[:1]:
        hx, wx = grid_size(x)
        hy, wy = grid_size(y)
        if hy % hx != 0 or wy % wx != 0:
            continue
        tr, tc = hy // hx, wy // wx
        if tr == 1 and tc == 1:
            continue

        def make_fn(tr=tr, tc=tc):
            return lambda g: tile_grid(g, len(g)*tr, len(g[0])*tc)

        prog = make_fn()
        if _fits_all(prog, pairs):
            return prog
    return None


def try_crop_to_content(pairs):
    bg = background(pairs[0][0])
    prog = lambda g: crop_to_content(g, bg)
    return prog if _fits_all(prog, pairs) else None


def try_histogram_barchart(pairs):
    """
    Count each non-background color and build a vertical bar chart.
    Covers tasks like b7999b51, f3cdc58f.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _barchart(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        hist = {c: 0 for c in colors_in(g) if c != bg}
        for row in g:
            for v in row:
                if v != bg:
                    hist[v] = hist.get(v, 0) + 1
        # Sort colors by count descending (or by color value)
        sorted_colors = sorted(hist.keys(), key=lambda c: (-hist[c], c))
        out = [[bg] * w for _ in range(h)]
        for col_idx, color in enumerate(sorted_colors):
            if col_idx >= w:
                break
            bar_h = hist[color]
            for r in range(h - bar_h, h):
                out[r][col_idx] = color
        return out

    return _barchart if _fits_all(_barchart, pairs) else None


def try_region_boolean(pairs):
    """
    Split grid on separators, apply boolean op (OR/AND/XOR) between two regions.
    Also tries recoloring the result (1→N) for tasks where output uses a specific color.
    Covers tasks like e133d23d, 0520fde7.
    """
    if not pairs:
        return None

    x0, y0 = pairs[0]
    sep_rows, sep_cols = find_separators(x0)
    if not sep_rows and not sep_cols:
        return None

    # Collect all colors in expected output
    out_colors = list(colors_in(y0) - {background(y0)}) + [1]

    for op_name, op in [("or", grid_or), ("and", grid_and), ("xor", grid_xor)]:
        for out_color in out_colors:
            def make_fn(sr=sep_rows, sc=sep_cols, op=op, oc=out_color):
                def fn(g):
                    regions = split_by_separators(g, sr, sc)
                    flat = [r for row in regions for r in row]
                    if len(flat) < 2:
                        return g
                    result = op(flat[0], flat[1])
                    # Recolor all non-zero cells to the target color
                    bg2 = background(result)
                    return [[oc if v != bg2 else bg2 for v in row] for row in result]
                return fn
            prog = make_fn()
            if _fits_all(prog, pairs):
                return prog
    return None


def try_separator_template_stamp(pairs):
    """
    Split on separator. One region = palette (NxN blocks → logical grid).
    Other region = template of marker positions.
    Stamp palette block wherever marker appears.
    Covers b4a43f3b, 12422b43-style tasks.
    """
    x0, _ = pairs[0]
    sep_rows, sep_cols = find_separators(x0)
    if not sep_rows and not sep_cols:
        return None

    def _try_stamp(sr, sc, x0, y0):
        regions = split_by_separators(x0, sr, sc)
        flat = [r for row in regions for r in row]
        if len(flat) < 2:
            return None
        # Try each region as palette, rest as template
        for pi in range(len(flat)):
            palette = flat[pi]
            template = flat[1 - pi] if len(flat) == 2 else None
            if template is None:
                continue
            ph, pw = grid_size(palette)
            th, tw = grid_size(template)
            # Try block sizes
            for block_h in range(1, ph + 1):
                if ph % block_h != 0:
                    continue
                for block_w in range(1, pw + 1):
                    if pw % block_w != 0:
                        continue
                    rows_p = ph // block_h
                    cols_p = pw // block_w
                    # Extract block palette
                    pal_grid = [
                        [palette[r * block_h][c * block_w] for c in range(cols_p)]
                        for r in range(rows_p)
                    ]
                    # Find marker color in template
                    bg_t = background(template)
                    markers = {c for row in template for c in row if c != bg_t}
                    for marker in markers:
                        def make_stamp_fn(pal_grid=pal_grid, template=template,
                                          marker=marker, block_h=block_h, block_w=block_w,
                                          th=th, tw=tw, sr=sr, sc=sc):
                            def fn(g):
                                regs = split_by_separators(g, sr, sc)
                                flat2 = [r for row in regs for r in row]
                                if len(flat2) < 2:
                                    return g
                                tmpl = flat2[1 - pi] if len(flat2) == 2 else flat2[1]
                                out_h = len(tmpl) * block_h
                                out_w = len(tmpl[0]) * block_w
                                out = empty_grid(out_h, out_w)
                                for tr in range(len(tmpl)):
                                    for tc in range(len(tmpl[0])):
                                        if tmpl[tr][tc] == marker:
                                            for pr in range(len(pal_grid)):
                                                for pc in range(len(pal_grid[0])):
                                                    out[tr * block_h + pr][tc * block_w + pc] = pal_grid[pr][pc]
                                return out
                            return fn

                        prog = make_stamp_fn()
                        if _fits_all(prog, pairs):
                            return prog
        return None

    return _try_stamp(sep_rows, sep_cols, x0, _)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Object-based recognizers
# ═══════════════════════════════════════════════════════════════════════════════

def try_shape_match_recolor(pairs):
    """
    Find objects of one color that match shapes of another color;
    recolor them to the matched color.
    Covers 2a5f8217.
    """
    if not _output_same_size_as_input(pairs):
        return None

    x0, y0 = pairs[0]
    bg = background(x0)

    # Find template color: objects whose shape appears twice (once as 1, once as color)
    all_objs = find_objects_excluding(x0, {bg})
    shape_to_colors: Dict = {}
    for obj in all_objs:
        shape = normalize_shape(obj)
        color = object_color(x0, obj)
        if color is None:
            continue
        shape_to_colors.setdefault(shape, set()).add(color)

    # Find pairs (placeholder_color, target_color) with matching shapes
    placeholder = None
    for shape, color_set in shape_to_colors.items():
        matching = [s for s, cs in shape_to_colors.items() if s == shape and cs != color_set]
        for other_shape in matching:
            for c1 in color_set:
                for c2 in (shape_to_colors.get(other_shape) or set()):
                    if c1 != c2:
                        placeholder = c1

    # If one color's shapes always match another color's shapes
    all_colors = list({c for row in x0 for c in row if c != bg})
    for candidate_ph in all_colors:
        def make_fn(ph=candidate_ph, bg=bg):
            def fn(g):
                bg2 = background(g)
                ph_objs = find_objects_of_color(g, ph)
                target_objs = find_objects_excluding(g, {bg2, ph})
                if not ph_objs or not target_objs:
                    return g
                # Build shape → color mapping from non-placeholder objects
                shape_map = {}
                for obj in target_objs:
                    c = object_color(g, obj)
                    if c is None:
                        continue
                    for orient in object_orientations(obj):
                        shape_map[orient] = c
                out = copy_grid(g)
                for obj in ph_objs:
                    for orient in object_orientations(obj):
                        if orient in shape_map:
                            for r, c in obj:
                                out[r][c] = shape_map[orient]
                            break
                return out
            return fn
        prog = make_fn()
        if _fits_all(prog, pairs):
            return prog
    return None


def try_stamp_by_mapping(pairs):
    """
    abc82100-style: find stamp-shape groups + color-chain pairs,
    then stamp each source cell with its mapped shape and target color.
    """
    if not _output_same_size_as_input(pairs):
        return None

    x0, y0 = pairs[0]
    all_colors = sorted(colors_in(x0) - {0})

    # Try each color as the "stamp template" color
    for stamp_color in all_colors:
        stamp_groups = find_objects_of_color(x0, stamp_color, connectivity=8)
        if not stamp_groups:
            continue

        # Try each other color as the "chain" background
        remaining_colors = [c for c in all_colors if c != stamp_color]
        chain_objs = []
        for obj in find_objects_excluding(x0, {0, stamp_color}):
            if len(obj) == 2 and len(object_colors(x0, obj)) == 2:
                chain_objs.append(obj)

        if not chain_objs:
            continue

        # Extract color mapping from chains
        color_map = extract_color_chains(x0, chain_objs)
        if not color_map:
            continue

        # Assign each chain to nearest stamp group
        assignments = assign_objects_to_nearest(chain_objs, stamp_groups)

        # Build: source_color → (target_color, offsets relative to chain's reference)
        chain_cells = frozenset(c for obj in chain_objs for c in obj)
        stamp_cells = frozenset(c for g in stamp_groups for c in g)

        # Determine offsets per assignment
        per_color_data: Dict = {}
        for chain, group_idx in zip(chain_objs, assignments):
            cells = list(chain)
            c1 = x0[cells[0][0]][cells[0][1]]
            c2 = x0[cells[1][0]][cells[1][1]]
            src, tgt = (c1, c2) if c2 in color_map.get(c1, {c2}) else (c2, c1)
            grp = stamp_groups[group_idx]
            ref = cells[0]  # reference point: use first chain cell
            offsets = frozenset((r - ref[0], c - ref[1]) for r, c in grp)
            per_color_data[src] = (tgt if tgt != src else color_map.get(src, src), offsets, ref)

        def make_fn(per_color_data=per_color_data, stamp_cells=stamp_cells,
                    chain_cells=chain_cells, stamp_color=stamp_color):
            def fn(g):
                h, w = grid_size(g)
                out = [[g[r][c] if (r,c) not in stamp_cells and (r,c) not in chain_cells
                        and g[r][c] not in per_color_data else 0
                        for c in range(w)] for r in range(h)]
                for r in range(h):
                    for c in range(w):
                        v = g[r][c]
                        if v in per_color_data and (r, c) not in chain_cells:
                            tgt, offsets, _ = per_color_data[v]
                            for dr, dc in offsets:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w:
                                    out[nr][nc] = tgt
                return out
            return fn

        prog = make_fn()
        if _fits_all(prog, pairs):
            return prog
    return None


def try_run_length_group(pairs):
    """
    Group consecutive identical rows into blocks; recolor every k-th block.
    Covers 22a4bbc2.
    """
    if not _output_same_size_as_input(pairs):
        return None

    for recolor_val in range(1, 10):
        for period in range(2, 6):
            def make_fn(period=period, rv=recolor_val):
                def fn(g):
                    rows = len(g)
                    out = [row[:] for row in g]
                    blocks = []
                    i = 0
                    while i < rows:
                        j = i + 1
                        while j < rows and g[j] == g[i]:
                            j += 1
                        blocks.append((i, j - 1))
                        i = j
                    for idx, (start, end) in enumerate(blocks):
                        if idx % period == 0:
                            for r in range(start, end + 1):
                                for c in range(len(g[r])):
                                    if out[r][c] != 0:
                                        out[r][c] = rv
                    return out
                return fn
            prog = make_fn()
            if _fits_all(prog, pairs):
                return prog
    return None


def try_frame_fill(pairs):
    """
    Find rectangular frames; fill their interior with a specific color.
    Covers b5ca7ac4-style tasks.
    """
    if not _output_same_size_as_input(pairs):
        return None

    x0, y0 = pairs[0]
    bg = background(x0)

    def _frame_fill(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg2 = background(g)
        out = copy_grid(g)
        # Find all 5-row×5-col frames (generalize to any size)
        for size in range(3, max(h, w)):
            for r in range(h - size + 1):
                for c in range(w - size + 1):
                    border_color = g[r][c]
                    if border_color == bg2:
                        continue
                    is_frame = True
                    for i in range(size):
                        for j in range(size):
                            is_border = (i == 0 or i == size-1 or j == 0 or j == size-1)
                            if is_border and g[r+i][c+j] != border_color:
                                is_frame = False
                                break
                        if not is_frame:
                            break
                    if is_frame:
                        # Fill interior
                        interior_color = g[r+1][c+1]
                        if interior_color != bg2 and interior_color != border_color:
                            for i in range(1, size - 1):
                                for j in range(1, size - 1):
                                    out[r+i][c+j] = interior_color
        return out

    return _frame_fill if _fits_all(_frame_fill, pairs) else None


def try_fractal_self_multiply(pairs):
    """
    Each non-zero cell (r,c) in the input becomes a full copy of the input
    placed at block position (r,c) in a tiled output.
    Covers 007bbfb7 and similar fractal/self-similar tasks.
    """
    x0, y0 = pairs[0]
    h, w = grid_size(x0)
    oh, ow = grid_size(y0)
    if oh != h * h or ow != w * w:
        # Try h*w vs w*h too (non-square)
        if oh != h * h or ow != w * w:
            pass  # still try below

    bg = background(x0)

    def _fractal(g: Grid) -> Grid:
        gh, gw = grid_size(g)
        out = empty_grid(gh * gh, gw * gw)
        for br in range(gh):
            for bc in range(gw):
                if g[br][bc] != bg:
                    for r in range(gh):
                        for c in range(gw):
                            out[br * gh + r][bc * gw + c] = g[r][c]
        return out

    # Also try: each non-zero cell → copy, scaled by non-zero color
    def _fractal_color(g: Grid) -> Grid:
        gh, gw = grid_size(g)
        out = empty_grid(gh * gh, gw * gw)
        for br in range(gh):
            for bc in range(gw):
                color = g[br][bc]
                if color != 0:
                    for r in range(gh):
                        for c in range(gw):
                            if g[r][c] != 0:
                                out[br * gh + r][bc * gw + c] = color
        return out

    for fn in [_fractal, _fractal_color]:
        if _fits_all(fn, pairs):
            return fn
    return None


def try_checkerboard_tile(pairs):
    """
    Tile input in NxM blocks; alternate rows or cols get flip_h / flip_v.
    Covers 00576224: 2x2 → 6x6 with alternating flip_h on row bands.
    """
    x0, y0 = pairs[0]
    hx, wx = grid_size(x0)
    hy, wy = grid_size(y0)
    if hy % hx != 0 or wy % wx != 0:
        return None
    tr, tc = hy // hx, wy // wx

    for row_flip in [flip_h, flip_v, None]:
        for col_flip in [flip_h, flip_v, None]:
            if row_flip is None and col_flip is None:
                continue

            def make_fn(tr=tr, tc=tc, rf=row_flip, cf=col_flip):
                def fn(g):
                    gh, gw = grid_size(g)
                    out = empty_grid(gh * tr, gw * tc)
                    for br in range(tr):
                        for bc in range(tc):
                            block = g
                            if rf is not None and br % 2 == 1:
                                block = rf(block)
                            if cf is not None and bc % 2 == 1:
                                block = cf(block)
                            for r in range(gh):
                                for c in range(gw):
                                    out[br * gh + r][bc * gw + c] = block[r][c]
                    return out
                return fn

            prog = make_fn()
            if _fits_all(prog, pairs):
                return prog
    return None


def try_column_height_rank(pairs):
    """
    Find vertical bars of a single color; rank by height (tallest=1).
    Recolor each bar by its rank.
    Covers 08ed6ac7.
    """
    if not _output_same_size_as_input(pairs):
        return None

    x0, y0 = pairs[0]
    bg = background(x0)
    fg_colors = list(colors_in(x0) - {bg})
    if len(fg_colors) != 1:
        return None

    bar_color = fg_colors[0]

    def _col_rank(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg2 = background(g)
        bar_c = next((c for c in colors_in(g) if c != bg2), None)
        if bar_c is None:
            return g
        # Measure height of each column's bar
        col_heights = {}
        for c in range(w):
            col = [g[r][c] for r in range(h)]
            cnt = sum(1 for v in col if v == bar_c)
            if cnt > 0:
                col_heights[c] = cnt
        if not col_heights:
            return g
        # Rank: unique heights sorted descending → rank 1,2,3,...
        unique_h = sorted(set(col_heights.values()), reverse=True)
        rank_of = {h: i+1 for i, h in enumerate(unique_h)}
        out = copy_grid(g)
        for c, ht in col_heights.items():
            rank = rank_of[ht]
            for r in range(h):
                if g[r][c] == bar_c:
                    out[r][c] = rank
        return out

    return _col_rank if _fits_all(_col_rank, pairs) else None


def try_row_height_rank(pairs):
    """Same as column_height_rank but for horizontal bars."""
    if not _output_same_size_as_input(pairs):
        return None

    x0, _ = pairs[0]
    bg = background(x0)
    fg = list(colors_in(x0) - {bg})
    if len(fg) != 1:
        return None

    def _row_rank(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg2 = background(g)
        bar_c = next((c for c in colors_in(g) if c != bg2), None)
        if bar_c is None:
            return g
        row_heights = {}
        for r in range(h):
            cnt = sum(1 for v in g[r] if v == bar_c)
            if cnt > 0:
                row_heights[r] = cnt
        if not row_heights:
            return g
        unique_h = sorted(set(row_heights.values()), reverse=True)
        rank_of = {h: i+1 for i, h in enumerate(unique_h)}
        out = copy_grid(g)
        for r, ht in row_heights.items():
            rank = rank_of[ht]
            for c in range(w):
                if g[r][c] == bar_c:
                    out[r][c] = rank
        return out

    return _row_rank if _fits_all(_row_rank, pairs) else None


def try_extract_region(pairs):
    """
    Split by separators; output = one of the sub-regions.
    Covers 0520fde7 (input 3x7 split by 5s → one 3x3 side).
    """
    x0, y0 = pairs[0]
    sep_rows, sep_cols = find_separators(x0)
    if not sep_rows and not sep_cols:
        return None

    regions_flat = [r for row in split_by_separators(x0, sep_rows, sep_cols) for r in row]
    for idx, region in enumerate(regions_flat):
        if grid_size(region) == grid_size(y0):
            def make_fn(sr=sep_rows, sc=sep_cols, i=idx):
                def fn(g):
                    flat = [r for row in split_by_separators(g, sr, sc) for r in row]
                    return flat[i] if i < len(flat) else g
                return fn
            prog = make_fn()
            if _fits_all(prog, pairs):
                return prog
    return None


def try_object_count_place(pairs):
    """
    Count objects per color; place output cells according to count/position.
    Covers counting-type tasks.
    """
    if not _output_same_size_as_input(pairs):
        return None
    return None


def try_gravity_down(pairs):
    """
    Each column: non-bg cells fall to the bottom (gravity down).
    Covers 1e0a9b12 and similar tasks.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _gravity_down(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        out = [[bg] * w for _ in range(h)]
        for c in range(w):
            col_vals = [g[r][c] for r in range(h) if g[r][c] != bg]
            for i, v in enumerate(col_vals):
                out[h - len(col_vals) + i][c] = v
        return out

    return _gravity_down if _fits_all(_gravity_down, pairs) else None


def try_gravity_up(pairs):
    """Each column: non-bg cells float to the top. Covers 03560426-style tasks."""
    if not _output_same_size_as_input(pairs):
        return None

    def _gravity_up(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        out = [[bg] * w for _ in range(h)]
        for c in range(w):
            col_vals = [g[r][c] for r in range(h) if g[r][c] != bg]
            for i, v in enumerate(col_vals):
                out[i][c] = v
        return out

    return _gravity_up if _fits_all(_gravity_up, pairs) else None


def try_gravity_left(pairs):
    """Each row: non-bg cells slide left."""
    if not _output_same_size_as_input(pairs):
        return None

    def _gravity_left(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        out = [[bg] * w for _ in range(h)]
        for r in range(h):
            row_vals = [v for v in g[r] if v != bg]
            for i, v in enumerate(row_vals):
                out[r][i] = v
        return out

    return _gravity_left if _fits_all(_gravity_left, pairs) else None


def try_gravity_right(pairs):
    """Each row: non-bg cells slide right."""
    if not _output_same_size_as_input(pairs):
        return None

    def _gravity_right(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        out = [[bg] * w for _ in range(h)]
        for r in range(h):
            row_vals = [v for v in g[r] if v != bg]
            for i, v in enumerate(row_vals):
                out[r][w - len(row_vals) + i] = v
        return out

    return _gravity_right if _fits_all(_gravity_right, pairs) else None


def try_fill_enclosed(pairs):
    """
    BFS flood-fill from the border; cells unreachable AND currently bg → fill with new color.
    Covers 00d62c1b (enclosed rectangles filled with color 4).
    """
    if not _output_same_size_as_input(pairs):
        return None

    x0, y0 = pairs[0]
    bg = background(x0)
    # Find what new color appears in output but not input
    new_colors = colors_in(y0) - colors_in(x0)
    if not new_colors:
        return None
    fill_c = next(iter(new_colors))

    def _fill_enclosed(g: Grid, fc=fill_c) -> Grid:
        h, w = grid_size(g)
        bg2 = background(g)
        # BFS from all border cells that are bg
        visited = [[False]*w for _ in range(h)]
        queue = deque()
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h-1 or c == 0 or c == w-1) and g[r][c] == bg2:
                    if not visited[r][c]:
                        visited[r][c] = True
                        queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == bg2:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        out = copy_grid(g)
        for r in range(h):
            for c in range(w):
                if g[r][c] == bg2 and not visited[r][c]:
                    out[r][c] = fc
        return out

    return _fill_enclosed if _fits_all(_fill_enclosed, pairs) else None


def try_color_key_table(pairs):
    """
    Top-left NxM cells form a color mapping table.
    The function is self-contained: it reads the key table from the INPUT
    grid at runtime (since each example has its own key table).
    Covers 0becf7df (bidirectional swap) and similar recoloring tasks.
    """
    if not _output_same_size_as_input(pairs):
        return None

    x0, _ = pairs[0]
    h, w = grid_size(x0)

    def _read_cmap(g, table_h, table_w, bidirectional):
        """Read key table from top-left table_h×table_w of g."""
        bg = background(g)
        cmap: dict = {}
        for r in range(table_h):
            for c in range(0, table_w - 1, 2):
                src = g[r][c]
                dst = g[r][c + 1]
                if src == bg or dst == bg:
                    continue
                cmap[src] = dst
                if bidirectional:
                    cmap[dst] = src
        return cmap

    def _apply_cmap(g, table_h, table_w, bidirectional):
        cmap = _read_cmap(g, table_h, table_w, bidirectional)
        if not cmap:
            return g
        h2, w2 = grid_size(g)
        out = copy_grid(g)
        tc = {(r, c) for r in range(table_h) for c in range(table_w)}
        for r in range(h2):
            for c in range(w2):
                if (r, c) in tc:
                    continue
                v = g[r][c]
                if v in cmap:
                    out[r][c] = cmap[v]
        return out

    for bidirectional in (False, True):
        for table_h in range(1, min(4, h)):
            for table_w in range(2, min(5, w)):  # need at least 2 cols for a pair
                def make_fn(th=table_h, tw=table_w, bi=bidirectional):
                    def fn(g):
                        return _apply_cmap(g, th, tw, bi)
                    return fn

                prog = make_fn()
                if _fits_all(prog, pairs):
                    return prog
    return None


def try_interior_fill(pairs):
    """
    Shapes made of 1-cells contain a single colored marker. Fill 'interior'
    cells — those whose all 8 neighbors are non-background — with the marker
    color of their connected component. Covers 09c534e7.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _interior_fill(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        out = copy_grid(g)

        # Build connected components and find each component's marker
        visited: dict = {}
        comp_marker: dict = {}
        cid = 0
        for sr in range(h):
            for sc in range(w):
                if g[sr][sc] == bg or (sr, sc) in visited:
                    continue
                queue = deque([(sr, sc)])
                visited[(sr, sc)] = cid
                marker = None
                while queue:
                    r, c = queue.popleft()
                    v = g[r][c]
                    if v != bg and v != 1:
                        marker = v
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != bg and (nr, nc) not in visited:
                            visited[(nr, nc)] = cid
                            queue.append((nr, nc))
                comp_marker[cid] = marker
                cid += 1

        # Fill interior cells (all 8 neighbors non-bg) with component marker
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if g[r][c] == bg:
                    continue
                marker = comp_marker.get(visited.get((r, c)))
                if marker is None:
                    continue
                if all(g[r + dr][c + dc] != bg
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                       if (dr, dc) != (0, 0)):
                    out[r][c] = marker
        return out

    return _interior_fill if _fits_all(_interior_fill, pairs) else None


def try_adjacent_recolor(pairs):
    """
    Recolor cells of color A that are 8-adjacent to any cell of color B → color C.
    Covers 14754a24 (5s adjacent to 4s → 2).
    """
    if not _output_same_size_as_input(pairs):
        return None

    x0, y0 = pairs[0]
    bg = background(x0)
    fg = [c for c in colors_in(x0) if c != bg]

    for A in fg:
        for B in fg:
            if A == B:
                continue
            for C in range(10):
                if C == A:
                    continue

                def make_fn(a=A, b=B, c=C):
                    def fn(g: Grid) -> Grid:
                        h, w = grid_size(g)
                        out = copy_grid(g)
                        for r in range(h):
                            for cc in range(w):
                                if g[r][cc] != a:
                                    continue
                                for dr in (-1, 0, 1):
                                    for dc in (-1, 0, 1):
                                        if dr == 0 and dc == 0:
                                            continue
                                        nr, nc = r + dr, cc + dc
                                        if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == b:
                                            out[r][cc] = c
                                            break
                                    else:
                                        continue
                                    break
                        return out
                    return fn

                prog = make_fn()
                if _fits_all(prog, pairs):
                    return prog
    return None


def try_complete_symmetry(pairs):
    """
    Complete a nearly-symmetric grid by mirroring non-bg cells across
    horizontal, vertical, or both axes — using either the grid center
    or the pattern bounding-box center as the axis. Covers 11852cab.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _apply_sym(g: Grid, use_h: bool, use_v: bool, use_pattern_center: bool) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        nz = [(r, c) for r in range(h) for c in range(w) if g[r][c] != bg]
        if not nz:
            return g
        if use_pattern_center:
            r_mid = (min(r for r, c in nz) + max(r for r, c in nz)) / 2
            c_mid = (min(c for r, c in nz) + max(c for r, c in nz)) / 2
        else:
            r_mid = (h - 1) / 2
            c_mid = (w - 1) / 2

        out = copy_grid(g)
        # Iterative fill: keep applying until stable (handles chained reflections)
        for _ in range(4):
            changed = False
            for r in range(h):
                for c in range(w):
                    if out[r][c] == bg:
                        continue
                    targets = []
                    if use_h:
                        mc = round(2 * c_mid - c)
                        if 0 <= mc < w:
                            targets.append((r, mc))
                    if use_v:
                        mr = round(2 * r_mid - r)
                        if 0 <= mr < h:
                            targets.append((mr, c))
                    if use_h and use_v:
                        mr = round(2 * r_mid - r)
                        mc = round(2 * c_mid - c)
                        if 0 <= mr < h and 0 <= mc < w:
                            targets.append((mr, mc))
                    for tr, tc2 in targets:
                        if out[tr][tc2] == bg:
                            out[tr][tc2] = out[r][c]
                            changed = True
            if not changed:
                break
        return out

    for use_pattern_center in (False, True):
        for (use_h, use_v) in ((True, False), (False, True), (True, True)):
            def make_fn(uh=use_h, uv=use_v, upc=use_pattern_center):
                def fn(g):
                    return _apply_sym(g, uh, uv, upc)
                return fn
            prog = make_fn()
            if _fits_all(prog, pairs):
                return prog
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Enumerative depth-1 sweep
# ═══════════════════════════════════════════════════════════════════════════════

def _enumerate_depth1(pairs: List[Tuple[Grid, Grid]], time_limit: float = 1.5) -> Optional[Program]:
    """Try every single primitive with all color arguments."""
    t0 = time.time()
    x0 = pairs[0][0]
    all_c = list(colors_in(x0))
    bg = background(x0)
    fg_colors = [c for c in all_c if c != bg]

    # Geometric transforms (already tried, but cheap to re-check)
    for fn in [flip_h, flip_v, flip_diag, flip_antidiag,
               lambda g: rotate(g, 1), lambda g: rotate(g, 2), lambda g: rotate(g, 3)]:
        if _fits_all(fn, pairs):
            return fn

    # Recolor single colors
    for src in fg_colors:
        for dst in range(10):
            if dst == src:
                continue
            def make_fn(s=src, d=dst):
                return lambda g: recolor(g, s, d)
            prog = make_fn()
            if _fits_all(prog, pairs):
                return prog
            if time.time() - t0 > time_limit:
                return None

    # Swap two colors
    for i, c1 in enumerate(fg_colors):
        for c2 in fg_colors[i+1:]:
            def make_fn(a=c1, b=c2):
                return lambda g: swap_colors(g, a, b)
            prog = make_fn()
            if _fits_all(prog, pairs):
                return prog

    # Color maps of up to 3 colors
    if len(fg_colors) <= 4:
        for perm in permutations(range(10), len(fg_colors)):
            cmap = dict(zip(fg_colors, perm))
            def make_fn(m=cmap):
                return lambda g: apply_color_map(g, m)
            prog = make_fn()
            if _fits_all(prog, pairs):
                return prog
            if time.time() - t0 > time_limit:
                return None

    return None


def try_diagonal_tile(pairs):
    """
    Output is a full-grid diagonal tiling: output[r][c] = diagonal_map[(r+c) % period].
    Period and map are extracted from the non-bg diagonal stripe in the input.
    Covers 05269061.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _derive_and_apply(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        diag_map: dict = {}
        for r in range(h):
            for c in range(w):
                v = g[r][c]
                if v != bg:
                    d = r + c
                    if d in diag_map and diag_map[d] != v:
                        raise ValueError("Inconsistent diagonal")
                    diag_map[d] = v
        if not diag_map:
            raise ValueError("No markers")
        period = len(set(diag_map.values()))
        if period < 2:
            raise ValueError("Need at least 2 colors for tiling")
        mod_map: dict = {}
        for d, v in diag_map.items():
            key = d % period
            if key in mod_map and mod_map[key] != v:
                raise ValueError("Inconsistent mod mapping")
            mod_map[key] = v
        if len(mod_map) != period:
            raise ValueError("Not all residues covered")
        return [[mod_map[(r + c) % period] for c in range(w)] for r in range(h)]

    return _derive_and_apply if _fits_all(_derive_and_apply, pairs) else None


def try_stripe_tiling(pairs):
    """
    Single-pixel colored markers define repeating stripes that tile the grid.
    Orientation is auto-detected per input: tall grid (h>w) → fill rows,
    wide grid (w>h) → fill cols. Markers must be equally spaced.
    Covers 0a938d79.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _apply(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        orient = "rows" if h > w else "cols"

        pos_color: dict = {}
        for r in range(h):
            for c in range(w):
                v = g[r][c]
                if v != bg:
                    pos = r if orient == "rows" else c
                    if pos in pos_color and pos_color[pos] != v:
                        raise ValueError("Multi-color position")
                    pos_color[pos] = v

        positions = sorted(pos_color.keys())
        if len(positions) < 2:
            raise ValueError("Need at least 2 markers")

        step = positions[1] - positions[0]
        if step <= 0:
            raise ValueError("Non-positive step")
        for k in range(1, len(positions)):
            if positions[k] - positions[k - 1] != step:
                raise ValueError("Irregular spacing")

        colors = [pos_color[p] for p in positions]
        n = len(positions)
        first_pos = positions[0]
        max_pos = h if orient == "rows" else w

        out = [[bg] * w for _ in range(h)]
        k = 0
        while True:
            pos_k = first_pos + k * step
            if pos_k >= max_pos:
                break
            color = colors[k % n]
            if orient == "rows":
                for c in range(w):
                    out[pos_k][c] = color
            else:
                for r in range(h):
                    out[r][pos_k] = color
            k += 1
        return out

    return _apply if _fits_all(_apply, pairs) else None


def try_gravity_toward_object(pairs):
    """
    Two objects in the grid: one slides toward the other along their shared axis
    until they are adjacent.  The axis is determined by bounding-box overlap:
      - col overlap → vertical movement
      - row overlap → horizontal movement
    Both orderings (which object is the mover) are tried.
    Covers 05f2a901.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def make_prog(mover_is_smaller: bool):
        def _apply(g: Grid) -> Grid:
            from collections import defaultdict
            h, w = grid_size(g)
            bg = background(g)
            color_cells: dict = defaultdict(list)
            for r in range(h):
                for c in range(w):
                    v = g[r][c]
                    if v != bg:
                        color_cells[v].append((r, c))
            if len(color_cells) != 2:
                raise ValueError("Need exactly 2 non-bg colors")

            colors = sorted(color_cells.keys(), key=lambda c: len(color_cells[c]))
            if mover_is_smaller:
                mover_color, anchor_color = colors[0], colors[1]
            else:
                mover_color, anchor_color = colors[1], colors[0]

            mcells = color_cells[mover_color]
            acells = color_cells[anchor_color]

            mr1 = min(r for r, c in mcells)
            mr2 = max(r for r, c in mcells)
            mc1 = min(c for r, c in mcells)
            mc2 = max(c for r, c in mcells)
            ar1 = min(r for r, c in acells)
            ar2 = max(r for r, c in acells)
            ac1 = min(c for r, c in acells)
            ac2 = max(c for r, c in acells)

            col_overlap = max(mc1, ac1) <= min(mc2, ac2)
            row_overlap = max(mr1, ar1) <= min(mr2, ar2)

            dr = dc = 0
            if col_overlap and not row_overlap:
                if ar1 > mr2:
                    dr = ar1 - 1 - mr2
                elif ar2 < mr1:
                    dr = ar2 + 1 - mr1
                else:
                    raise ValueError("Objects already overlap vertically")
            elif row_overlap and not col_overlap:
                if ac1 > mc2:
                    dc = ac1 - 1 - mc2
                elif ac2 < mc1:
                    dc = ac2 + 1 - mc1
                else:
                    raise ValueError("Objects already overlap horizontally")
            else:
                raise ValueError("Cannot determine movement axis")

            out = [list(row) for row in g]
            for r, c in mcells:
                out[r][c] = bg
            for r, c in mcells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr][nc] = mover_color
            return out
        return _apply

    for flag in (True, False):
        prog = make_prog(flag)
        if _fits_all(prog, pairs):
            return prog
    return None


def try_block_tile_down(pairs):
    """
    Detect a top-aligned header column and tile the rows above its height downward,
    excluding the header column itself. Covers 12422b43-style tasks.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _detect_header(g: Grid) -> Tuple[int, int]:
        h, w = grid_size(g)
        bg = background(g)
        best: Optional[Tuple[int, int]] = None
        for c in range(w):
            if h < 2 or g[0][c] == bg:
                continue
            header_color = g[0][c]
            run_len = 1
            while run_len < h and g[run_len][c] == header_color:
                run_len += 1
            if run_len < 1:
                continue
            if any(g[r][c] != bg for r in range(run_len, h)):
                continue
            if best is None or run_len > best[1]:
                best = (c, run_len)
        if best is None:
            raise ValueError("No header column")
        return best

    def _apply(g: Grid) -> Grid:
        h, w = grid_size(g)
        bg = background(g)
        header_col, tile_height = _detect_header(g)
        last_content_row = -1
        for r in range(h):
            if any(g[r][c] != bg for c in range(w) if c != header_col):
                last_content_row = r
        if last_content_row < 0:
            raise ValueError("No content outside header column")
        tile_start = last_content_row + 1
        if tile_start >= h:
            raise ValueError("No space to tile downward")

        out = copy_grid(g)
        for r in range(tile_start, h):
            src_r = (r - tile_start) % tile_height
            for c in range(w):
                if c == header_col:
                    continue
                out[r][c] = g[src_r][c]
        return out

    return _apply if _fits_all(_apply, pairs) else None


def try_small_component_recolor(pairs):
    """
    Replace every same-color 4-connected component of size <= threshold with
    a fixed replacement color.  Covers 12eac192-style tasks where isolated /
    small-group pixels become color 3.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _components(g, bg):
        h, w = grid_size(g)
        visited = [[False] * w for _ in range(h)]
        comps = []
        for sr in range(h):
            for sc in range(w):
                if g[sr][sc] == bg or visited[sr][sc]:
                    continue
                col = g[sr][sc]
                stack = [(sr, sc)]
                cells = []
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= h or c < 0 or c >= w:
                        continue
                    if visited[r][c] or g[r][c] != col:
                        continue
                    visited[r][c] = True
                    cells.append((r, c))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        stack.append((r + dr, c + dc))
                comps.append((col, cells))
        return comps

    def _make_prog(threshold, replace_color):
        def _apply(g):
            bg = background(g)
            comps = _components(g, bg)
            out = copy_grid(g)
            for col, cells in comps:
                if len(cells) <= threshold:
                    for r, c in cells:
                        out[r][c] = replace_color
            return out
        return _apply

    # Try threshold 1 and 2 with replacement colors found in training outputs
    out_colors = set()
    for _, y in pairs:
        for row in y:
            for v in row:
                out_colors.add(v)
    in_colors = set()
    for x, _ in pairs:
        for row in x:
            for v in row:
                in_colors.add(v)
    new_colors = out_colors - in_colors
    # candidate replace colors: colors that appear in output but not input, or just try 3
    candidates = list(new_colors) if new_colors else []
    # also try colors that increase most in output
    for rc in [3, 2, 1, 4, 6, 7, 8, 9]:
        if rc not in candidates:
            candidates.append(rc)

    for threshold in (1, 2):
        for rc in candidates:
            prog = _make_prog(threshold, rc)
            if _fits_all(prog, pairs):
                return prog
    return None


def try_connect_diagonal(pairs):
    """
    For each pair of same-colored non-background pixels, draw a diagonal line
    (|dr| == |dc|) between them.  Covers 1f876c06-style tasks.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _apply(g):
        bg = background(g)
        h, w = grid_size(g)
        # Collect all non-bg pixels grouped by color
        from collections import defaultdict
        color_cells = defaultdict(list)
        for r in range(h):
            for c in range(w):
                if g[r][c] != bg:
                    color_cells[g[r][c]].append((r, c))
        out = copy_grid(g)
        for col, cells in color_cells.items():
            if len(cells) != 2:
                continue
            (r1, c1), (r2, c2) = cells
            dr = r2 - r1
            dc = c2 - c1
            if abs(dr) != abs(dc):
                raise ValueError(f"Not diagonal: {cells}")
            steps = abs(dr)
            sr = 1 if dr > 0 else -1
            sc = 1 if dc > 0 else -1
            for i in range(steps + 1):
                out[r1 + i * sr][c1 + i * sc] = col
        return out

    if _fits_all(_apply, pairs):
        return _apply

    # Also try with >2 cells per color: connect nearest pair
    def _apply_nearest(g):
        bg = background(g)
        h, w = grid_size(g)
        from collections import defaultdict
        color_cells = defaultdict(list)
        for r in range(h):
            for c in range(w):
                if g[r][c] != bg:
                    color_cells[g[r][c]].append((r, c))
        out = copy_grid(g)
        for col, cells in color_cells.items():
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    r1, c1 = cells[i]
                    r2, c2 = cells[j]
                    dr = r2 - r1
                    dc = c2 - c1
                    if abs(dr) != abs(dc):
                        continue
                    steps = abs(dr)
                    sr = 1 if dr > 0 else -1
                    sc = 1 if dc > 0 else -1
                    for k in range(steps + 1):
                        out[r1 + k * sr][c1 + k * sc] = col
        return out

    if _fits_all(_apply_nearest, pairs):
        return _apply_nearest
    return None


def try_rectangle_corner_mark(pairs):
    """
    Per connected-component of each non-bg color: if the component's bounding
    box is a square (side >= 2) AND all perimeter cells of that bbox are the
    component's color, mark 8 orthogonal exterior corner positions with marker
    color.  Uses per-component detection to avoid spurious sub-square detection
    inside filled rectangular blocks.  Covers 14b8e18c-style tasks.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _connected_components(g, color):
        h, w = grid_size(g)
        visited = [[False] * w for _ in range(h)]
        comps = []
        for sr in range(h):
            for sc in range(w):
                if g[sr][sc] == color and not visited[sr][sc]:
                    comp = []
                    stack = [(sr, sc)]
                    visited[sr][sc] = True
                    while stack:
                        r, c = stack.pop()
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and g[nr][nc]==color:
                                visited[nr][nc] = True
                                stack.append((nr, nc))
                    comps.append(comp)
        return comps

    def _square_rects_from_components(g, bg):
        h, w = grid_size(g)
        seen_colors = set(g[r][c] for r in range(h) for c in range(w) if g[r][c] != bg)
        rects = []
        for color in seen_colors:
            for comp in _connected_components(g, color):
                r1 = min(r for r, c in comp)
                r2 = max(r for r, c in comp)
                c1 = min(c for r, c in comp)
                c2 = max(c for r, c in comp)
                if r2 - r1 < 1:          # single row — skip
                    continue
                if r2 - r1 != c2 - c1:   # must be square
                    continue
                comp_set = set(comp)
                # All perimeter cells of bbox must be this color
                top = all((r1, c) in comp_set for c in range(c1, c2+1))
                bot = all((r2, c) in comp_set for c in range(c1, c2+1))
                lft = all((r, c1) in comp_set for r in range(r1, r2+1))
                rgt = all((r, c2) in comp_set for r in range(r1, r2+1))
                if top and bot and lft and rgt:
                    rects.append((r1, r2, c1, c2))
        return rects

    def _make_prog(marker_color):
        def _apply(g):
            bg = background(g)
            h, w = grid_size(g)
            out = copy_grid(g)
            for r1, r2, c1, c2 in _square_rects_from_components(g, bg):
                for er, ec in [(r1-1,c1),(r1-1,c2),(r2+1,c1),(r2+1,c2),
                               (r1,c1-1),(r1,c2+1),(r2,c1-1),(r2,c2+1)]:
                    if 0 <= er < h and 0 <= ec < w:
                        out[er][ec] = marker_color
            return out
        return _apply

    in_all = set(v for x, _ in pairs for row in x for v in row)
    out_all = set(v for _, y in pairs for row in y for v in row)
    new_colors = list(out_all - in_all)
    marker_candidates = new_colors if new_colors else [2, 3, 4]

    for mc in marker_candidates + [2, 3, 4]:
        prog = _make_prog(mc)
        if _fits_all(prog, pairs):
            return prog
    return None


def try_color_decoration(pairs):
    """
    Each non-background source color gets a fixed decoration: stamp a set of
    neighbor offsets with a specific decoration color.
    E.g. color 1 → add color 7 at orthogonal neighbors; color 2 → add color 4
    at diagonal neighbors.  Covers 0ca9ddb6-style tasks.
    """
    if not _output_same_size_as_input(pairs):
        return None

    ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ALL8 = ORTHO + DIAG

    def _make_prog(decorations):
        # decorations: list of (src_color, offsets, deco_color)
        def _apply(g):
            bg = background(g)
            h, w = grid_size(g)
            out = copy_grid(g)
            for r in range(h):
                for c in range(w):
                    col = g[r][c]
                    if col == bg:
                        continue
                    for src_col, offsets, deco_col in decorations:
                        if col != src_col:
                            continue
                        for dr, dc in offsets:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and g[nr][nc] == bg:
                                out[nr][nc] = deco_col
            return out
        return _apply

    bg0 = background(pairs[0][0])
    src_colors = sorted(set(
        v for x, _ in pairs for row in x for v in row if v != bg0
    ))

    # For each src color figure out which offsets and deco color from train data
    offset_sets = [ORTHO, DIAG, ALL8]
    # Try enumerating decorations per src color
    # Build candidate (offsets, deco_color) for each src color by looking at diffs
    deco_candidates_per_src = {}
    for sc in src_colors:
        cands = []
        # Find where sc appears in inputs, what's added in outputs
        for x, y in pairs:
            h, w = grid_size(x)
            for r in range(h):
                for c in range(w):
                    if x[r][c] == sc:
                        for dr, dc in ALL8:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if x[nr][nc] == bg0 and y[nr][nc] != bg0:
                                    cands.append((dr, dc, y[nr][nc]))
        if not cands:
            continue
        # Find most common (offset, deco_color) pattern
        from collections import Counter
        counts = Counter(cands)
        # Group by deco_color
        by_deco = {}
        for (dr, dc, deco_col), cnt in counts.items():
            by_deco.setdefault(deco_col, []).append((dr, dc))
        deco_candidates_per_src[sc] = by_deco

    if not deco_candidates_per_src:
        return None

    # Build all combinations
    def _build_combos(src_list):
        if not src_list:
            yield []
            return
        sc = src_list[0]
        rest = src_list[1:]
        if sc not in deco_candidates_per_src:
            for combo in _build_combos(rest):
                yield combo
            return
        for deco_col, offsets in deco_candidates_per_src[sc].items():
            for combo in _build_combos(rest):
                yield [(sc, offsets, deco_col)] + combo

    for decorations in _build_combos(src_colors):
        if not decorations:
            continue
        prog = _make_prog(decorations)
        if _fits_all(prog, pairs):
            return prog
    return None


def try_plus_expand(pairs):
    """
    Find a plus/cross shape (center + 4 identical orthogonal arms of length 1,
    no other non-bg neighbors); output doubles arm length to 2 and fills the
    4 diagonal cells at dist 1 AND dist 2 with the center color.
    Covers 0962bcdd-style tasks.
    """
    if not _output_same_size_as_input(pairs):
        return None

    def _find_plus(g, bg):
        h, w = grid_size(g)
        results = []
        for r in range(2, h - 2):
            for c in range(2, w - 2):
                center_col = g[r][c]
                if center_col == bg:
                    continue
                arm_positions = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                arm_colors = [g[ar][ac] for ar,ac in arm_positions]
                # All 4 arms must be same non-bg color
                if any(a == bg for a in arm_colors):
                    continue
                if len(set(arm_colors)) > 1:
                    continue
                arm_col = arm_colors[0]
                # No existing content at dist-2 arms or dist-1 diagonals
                clear = True
                for dr, dc in [(-2,0),(2,0),(0,-2),(0,2),
                                (-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != bg:
                        clear = False
                        break
                if not clear:
                    continue
                results.append((r, c, center_col, arm_col))
        return results

    def _apply(g):
        bg = background(g)
        h, w = grid_size(g)
        out = [list(row) for row in g]
        pluses = _find_plus(g, bg)
        for cr, cc, center_col, arm_col in pluses:
            # Erase original length-1 arms
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                out[cr+dr][cc+dc] = bg
            # Draw extended arms (length 1 and 2)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                for dist in (1, 2):
                    nr, nc = cr + dr*dist, cc + dc*dist
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr][nc] = arm_col
            # Fill diagonal cells at dist 1 AND dist 2 with center color
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                for dist in (1, 2):
                    nr, nc = cr + dr*dist, cc + dc*dist
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr][nc] = center_col
        return out

    if _fits_all(_apply, pairs):
        return _apply
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Synthesizer entry point
# ═══════════════════════════════════════════════════════════════════════════════

# Ordered strategy registry: (name, fn, max_time_s)
_STRATEGIES = [
    ("identity",               try_identity,               0.001),
    ("geometric_transform",    try_geometric_transforms,   0.05),
    ("color_mapping",          try_color_mapping,          0.1),
    ("invert_tile",            try_invert_and_tile,        0.05),
    ("tiling",                 try_tiling,                 0.05),
    ("checkerboard_tile",      try_checkerboard_tile,      0.1),
    ("fractal_self_multiply",  try_fractal_self_multiply,  0.1),
    ("crop_content",           try_crop_to_content,        0.05),
    ("extract_region",         try_extract_region,         0.2),
    ("histogram_barchart",     try_histogram_barchart,     0.1),
    ("column_height_rank",     try_column_height_rank,     0.1),
    ("row_height_rank",        try_row_height_rank,        0.1),
    ("diagonal_tile",          try_diagonal_tile,          0.05),
    ("stripe_tiling",          try_stripe_tiling,          0.1),
    ("block_tile_down",        try_block_tile_down,        0.3),
    ("gravity_toward_object",  try_gravity_toward_object,  0.1),
    ("small_component_recolor",try_small_component_recolor,0.3),
    ("connect_diagonal",       try_connect_diagonal,       0.2),
    ("rectangle_corner_mark",  try_rectangle_corner_mark,  0.5),
    ("color_decoration",       try_color_decoration,       0.5),
    ("plus_expand",            try_plus_expand,            0.3),
    ("gravity_down",           try_gravity_down,           0.05),
    ("gravity_up",             try_gravity_up,             0.05),
    ("gravity_left",           try_gravity_left,           0.05),
    ("gravity_right",          try_gravity_right,          0.05),
    ("fill_enclosed",          try_fill_enclosed,          0.1),
    ("interior_fill",          try_interior_fill,          0.3),
    ("adjacent_recolor",       try_adjacent_recolor,       0.5),
    ("complete_symmetry",      try_complete_symmetry,      0.1),
    ("color_key_table",        try_color_key_table,        0.2),
    ("region_boolean",         try_region_boolean,         0.3),
    ("separator_template",     try_separator_template_stamp, 0.5),
    ("shape_match_recolor",    try_shape_match_recolor,    0.5),
    ("stamp_by_mapping",       try_stamp_by_mapping,       1.0),
    ("run_length_group",       try_run_length_group,       0.3),
    ("frame_fill",             try_frame_fill,             0.5),
    ("enumerate_depth1",       _enumerate_depth1,          2.0),
]


class Synthesizer:
    """
    Program synthesizer for ARC-AGI tasks.

    Usage
    -----
        syn = Synthesizer()
        prog = syn.synthesize(train_pairs)
        if prog:
            prediction = prog(test_input)
    """

    def __init__(self, time_budget: float = 5.0):
        self.time_budget = time_budget

    def synthesize(
        self,
        pairs: List[Tuple[Grid, Grid]],
        verbose: bool = False,
    ) -> Optional[Program]:
        """
        Find a program P such that P(x) == y for all (x,y) in pairs.
        Returns None if no program found within time_budget.
        """
        if not pairs:
            return None

        t0 = time.time()
        remaining = self.time_budget

        for name, strategy_fn, max_t in _STRATEGIES:
            if remaining <= 0:
                break
            try:
                prog = strategy_fn(pairs)
                if prog is not None:
                    if verbose:
                        print(f"  [Synthesizer] solved by: {name} "
                              f"({time.time()-t0:.3f}s)")
                    return prog
            except Exception:
                pass
            remaining = self.time_budget - (time.time() - t0)

        if verbose:
            print(f"  [Synthesizer] no program found "
                  f"({time.time()-t0:.3f}s)")
        return None

    def solve_task(self, task: dict, verbose: bool = False) -> Optional[List[List[int]]]:
        """
        Convenience method: synthesize from train pairs, apply to test input.
        Returns the predicted output grid, or None.
        """
        pairs = [(ex["input"], ex["output"]) for ex in task.get("train", [])]
        test_input = task["test"][0]["input"]

        prog = self.synthesize(pairs, verbose=verbose)
        if prog is None:
            return None
        try:
            return prog(test_input)
        except Exception:
            return None
