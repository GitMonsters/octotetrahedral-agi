"""
RE-ARC auto-solver: tries a battery of primitive transforms against each task's
training pairs, picks the best, and produces a submission JSON.

Usage:
    python3 -m arc3.re_arc_solver \
        --challenges /path/to/re-arc_test_challenges.json \
        --output submission_re_arc.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from itertools import product
from typing import Callable, Optional

logger = logging.getLogger("arc3.re_arc_solver")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def hflip(g: Grid) -> Grid:
    return [row[::-1] for row in g]

def vflip(g: Grid) -> Grid:
    return g[::-1]

def rot90(g: Grid) -> Grid:
    return [list(row) for row in zip(*g[::-1])]

def rot180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]

def rot270(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)][::-1]

def transpose(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)]

def bg_color(g: Grid) -> int:
    flat = [c for row in g for c in row]
    return Counter(flat).most_common(1)[0][0]

def bounding_box(g: Grid, exclude: int | None = None) -> tuple[int,int,int,int]:
    """Returns (r0, r1, c0, c1) inclusive bounding box of non-exclude cells."""
    h, w = len(g), len(g[0])
    bg = exclude if exclude is not None else bg_color(g)
    rows = [r for r in range(h) if any(g[r][c] != bg for c in range(w))]
    cols = [c for c in range(w) if any(g[r][c] != bg for r in range(h))]
    if not rows or not cols:
        return 0, h-1, 0, w-1
    return rows[0], rows[-1], cols[0], cols[-1]

def crop(g: Grid, r0: int, r1: int, c0: int, c1: int) -> Grid:
    return [row[c0:c1+1] for row in g[r0:r1+1]]

def crop_to_objects(g: Grid) -> Grid:
    r0, r1, c0, c1 = bounding_box(g)
    return crop(g, r0, r1, c0, c1)

def apply_color_map(g: Grid, cmap: dict[int,int]) -> Grid:
    return [[cmap.get(c, c) for c in row] for row in g]

def scale_up(g: Grid, k: int) -> Grid:
    return [[c for c in row for _ in range(k)] for row in g for _ in range(k)]

def fill_bg(g: Grid, fill: int) -> Grid:
    bg = bg_color(g)
    return [[fill if c == bg else c for c in row] for row in g]

def gravity(g: Grid, direction: str) -> Grid:
    """Slide non-background cells toward direction ('U','D','L','R')."""
    bg = bg_color(g)
    h, w = len(g), len(g[0])
    result = [[bg]*w for _ in range(h)]

    if direction in ('D', 'U'):
        for c in range(w):
            col = [g[r][c] for r in range(h)]
            nonbg = [v for v in col if v != bg]
            n = len(nonbg)
            if direction == 'D':
                filled = [bg]*(h-n) + nonbg
            else:
                filled = nonbg + [bg]*(h-n)
            for r in range(h):
                result[r][c] = filled[r]
    else:
        for r in range(h):
            row = g[r]
            nonbg = [v for v in row if v != bg]
            n = len(nonbg)
            if direction == 'R':
                filled = [bg]*(w-n) + nonbg
            else:
                filled = nonbg + [bg]*(w-n)
            result[r] = filled
    return result

def color_sort_rows(g: Grid) -> Grid:
    """Sort rows by their dominant non-bg color."""
    bg = bg_color(g)
    def row_key(row):
        c = Counter(v for v in row if v != bg)
        return c.most_common(1)[0][0] if c else bg
    return sorted(g, key=row_key)

def recolor_by_size(g: Grid) -> Grid:
    """Recolor connected components by their size (smaller=one color, larger=another)."""
    from collections import deque
    bg = bg_color(g)
    h, w = len(g), len(g[0])
    visited = [[False]*w for _ in range(h)]
    components = []
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and g[r][c] != bg:
                # BFS
                color = g[r][c]
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append((color, comp))
    if not components:
        return g
    sizes = sorted(set(len(c[1]) for c in components))
    result = [row[:] for row in g]
    if len(sizes) >= 2:
        small_size = sizes[0]
        # Give each unique size a color index
        for color, comp in components:
            new_c = color if len(comp) != small_size else (color + 1) % 10
            for r, c in comp:
                result[r][c] = new_c
    return result

def symmetry_complete(g: Grid) -> Grid:
    """Fill asymmetric cells using 4-fold symmetry (only for small grids ≤10x10)."""
    h, w = len(g), len(g[0])
    if h > 10 or w > 10:
        return g  # skip on large grids
    bg = bg_color(g)
    result = [row[:] for row in g]
    for r in range(h // 2 + 1):
        for c in range(w // 2 + 1):
            mirrors = [(r, w-1-c), (h-1-r, c), (h-1-r, w-1-c)]
            vals = [g[r][c]] + [g[mr][mc] for mr, mc in mirrors]
            non_bg = [v for v in vals if v != bg]
            if non_bg:
                fill = Counter(non_bg).most_common(1)[0][0]
                result[r][c] = fill
                for mr, mc in mirrors:
                    result[mr][mc] = fill
    return result

def count_and_fill(g: Grid) -> Grid:
    """Replace each object with a color equal to its size (mod 10). Skips grids > 20x20."""
    h, w = len(g), len(g[0])
    if h * w > 400:
        return g  # too large to be useful
    from collections import deque
    bg = bg_color(g)
    result = [row[:] for row in g]
    visited = [[False]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and g[r][c] != bg:
                color = g[r][c]
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                new_c = len(comp) % 10
                for cr, cc in comp:
                    result[cr][cc] = new_c
    return result

# ---------------------------------------------------------------------------
# Triangular tiling (task type 6ea14baa)
# ---------------------------------------------------------------------------

def triangular_tile(g: Grid) -> Grid:
    """
    Scale factor K = number of background cells.
    Output is K*H × K*W.
    Only runs if K ≤ 10 and output ≤ 30x30 (guard against huge allocations).
    """
    h, w = len(g), len(g[0])
    bg = bg_color(g)
    flat = [c for row in g for c in row]
    K = flat.count(bg)
    if K < 1 or K > 10 or K * h > 40 or K * w > 40:
        return g
    out_h, out_w = K * h, K * w
    out = [[bg] * out_w for _ in range(out_h)]

    # Place copy at (rb, cb) if rb + cb >= K - 1  (triangular lower-right)
    for rb in range(K):
        for cb in range(K):
            if rb + cb >= K - 1:
                for r in range(h):
                    for c in range(w):
                        out[rb*h + r][cb*w + c] = g[r][c]
    return out


# ---------------------------------------------------------------------------
# Learn color map from train pairs
# ---------------------------------------------------------------------------

def learn_color_map(pairs: list[tuple[Grid, Grid]]) -> Optional[dict[int,int]]:
    """Try to learn a consistent per-color mapping from all train pairs."""
    cmap: dict[int,int] = {}
    for inp, out in pairs:
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        h, w = len(inp), len(inp[0])
        for r in range(h):
            for c in range(w):
                k, v = inp[r][c], out[r][c]
                if k in cmap:
                    if cmap[k] != v:
                        return None
                else:
                    cmap[k] = v
    return cmap if cmap else None


# ---------------------------------------------------------------------------
# Primitive solver registry
# ---------------------------------------------------------------------------

def make_fixed(fn: Callable[[Grid], Grid]) -> Callable:
    def solver(inp: Grid, _pairs) -> Grid:
        return fn(inp)
    return solver

def make_with_cmap(cmap: dict[int,int]) -> Callable:
    def solver(inp: Grid, _pairs) -> Grid:
        return apply_color_map(inp, cmap)
    return solver

def make_gravity(d: str) -> Callable:
    def solver(inp: Grid, _pairs) -> Grid:
        return gravity(inp, d)
    return solver

def make_scale_up(k: int) -> Callable:
    def solver(inp: Grid, _pairs) -> Grid:
        return scale_up(inp, k)
    return solver

def make_crop_then(fn: Callable[[Grid], Grid]) -> Callable:
    def solver(inp: Grid, _pairs) -> Grid:
        return fn(crop_to_objects(inp))
    return solver

FIXED_PRIMITIVES: list[tuple[str, Callable[[Grid], Grid]]] = [
    ("identity",        lambda g: g),
    ("hflip",           hflip),
    ("vflip",           vflip),
    ("rot90",           rot90),
    ("rot180",          rot180),
    ("rot270",          rot270),
    ("transpose",       transpose),
    ("crop",            crop_to_objects),
    ("grav_D",          lambda g: gravity(g, 'D')),
    ("grav_U",          lambda g: gravity(g, 'U')),
    ("grav_L",          lambda g: gravity(g, 'L')),
    ("grav_R",          lambda g: gravity(g, 'R')),
    ("grav_D_hflip",    lambda g: hflip(gravity(g, 'D'))),
    ("grav_U_hflip",    lambda g: hflip(gravity(g, 'U'))),
    ("grav_D_vflip",    lambda g: vflip(gravity(g, 'D'))),
    ("triangular_tile", triangular_tile),
    ("count_fill",      count_and_fill),
    ("sym_complete",    symmetry_complete),
    ("scale2",          lambda g: scale_up(g, 2)),
    ("scale3",          lambda g: scale_up(g, 3)),
    ("scale4",          lambda g: scale_up(g, 4)),
    ("crop_rot90",      lambda g: rot90(crop_to_objects(g))),
    ("crop_rot180",     lambda g: rot180(crop_to_objects(g))),
    ("crop_hflip",      lambda g: hflip(crop_to_objects(g))),
    ("crop_vflip",      lambda g: vflip(crop_to_objects(g))),
]

# Maximum grid cells to allow primitive comparison (avoid huge grids)
_MAX_CELLS = 1600  # 40×40


# ---------------------------------------------------------------------------
# Task solver: tries all primitives + learned color map
# ---------------------------------------------------------------------------

def pairs_from_task(task: dict) -> list[tuple[Grid, Grid]]:
    return [(p['input'], p['output']) for p in task['train']]

def solve_task(task: dict) -> tuple[str, Optional[Callable]]:
    """
    Returns (method_name, solver_fn) or ("failed", None).
    solver_fn(inp, pairs) -> Grid
    """
    pairs = pairs_from_task(task)
    inp0, exp0 = pairs[0]
    n_cells_in = len(inp0) * len(inp0[0])
    n_cells_out = len(exp0) * len(exp0[0])
    large = n_cells_in > _MAX_CELLS or n_cells_out > _MAX_CELLS
    expensive = {"sym_complete", "count_fill", "triangular_tile",
                 "scale2", "scale3", "scale4"}

    # 1. Try fixed primitives
    for name, fn in FIXED_PRIMITIVES:
        if large and name in expensive:
            continue
        try:
            results = [fn(inp) for inp, _ in pairs]
        except Exception:
            continue
        if any(len(r) != len(e) or (r and len(r[0]) != len(e[0]))
               for r, (_, e) in zip(results, pairs)):
            continue
        if all(r == e for r, (_, e) in zip(results, pairs)):
            return name, make_fixed(fn)

    # 2. Learned color map
    cmap = learn_color_map(pairs)
    if cmap and all(apply_color_map(inp, cmap) == exp for inp, exp in pairs):
        return "color_map", make_with_cmap(cmap)

    # 3. Color map after crop
    try:
        crop_pairs = [(crop_to_objects(inp), exp) for inp, exp in pairs]
        if all(len(a) == len(e) and len(a[0]) == len(e[0]) for a, e in crop_pairs):
            cmap2 = learn_color_map(crop_pairs)
            if cmap2 and all(apply_color_map(crop_to_objects(inp), cmap2) == exp
                             for inp, exp in pairs):
                _cm2 = cmap2
                def _s2(inp, _, m=_cm2): return apply_color_map(crop_to_objects(inp), m)
                return "crop+color_map", _s2
    except Exception:
        pass

    # 4. Color map after simple primitives (skip on large grids)
    if not large:
        for name, fn in FIXED_PRIMITIVES[:8]:
            try:
                xpairs = [(fn(inp), exp) for inp, exp in pairs]
                if any(len(a) != len(e) or len(a[0]) != len(e[0]) for a, e in xpairs):
                    continue
                cmap3 = learn_color_map(xpairs)
                if cmap3 and all(apply_color_map(fn(inp), cmap3) == exp
                                 for inp, exp in pairs):
                    _fn3, _cm3 = fn, cmap3
                    def _s3(inp, _, f=_fn3, m=_cm3): return apply_color_map(f(inp), m)
                    return f"{name}+color_map", _s3
            except Exception:
                pass

    # 5. Double crop
    try:
        cc = [(crop_to_objects(crop_to_objects(inp)), exp) for inp, exp in pairs]
        if all(a == e for a, e in cc):
            return "double_crop", make_fixed(lambda g: crop_to_objects(crop_to_objects(g)))
    except Exception:
        pass

    # 6. Variable scale from ratio
    try:
        ih0, iw0 = len(inp0), len(inp0[0])
        oh0, ow0 = len(exp0), len(exp0[0])
        if oh0 % ih0 == 0 and ow0 % iw0 == 0 and oh0 // ih0 == ow0 // iw0:
            k = oh0 // ih0
            if k > 1 and all(scale_up(inp, k) == exp for inp, exp in pairs):
                return f"scale{k}", make_scale_up(k)
    except Exception:
        pass

    # 7. Triangular tile
    try:
        if all(triangular_tile(inp) == exp for inp, exp in pairs):
            return "triangular_tile", make_fixed(triangular_tile)
    except Exception:
        pass

    return "failed", None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_solver(challenges_path: str, output_path: str) -> None:
    with open(challenges_path) as f:
        challenges = json.load(f)

    submission: dict[str, list[list[list[int]]]] = {}
    stats: dict[str, int] = Counter()

    for tid, task in challenges.items():
        method, solver_fn = solve_task(task)
        stats[method] += 1
        train_pairs = pairs_from_task(task)

        preds: list[list[list[int]]] = []
        for p in task['test']:
            inp = p['input']
            if solver_fn is not None:
                try:
                    pred = solver_fn(inp, train_pairs)
                except Exception as e:
                    logger.warning(f"{tid}: solver error: {e}")
                    pred = inp  # fallback: return input unchanged
            else:
                # Fallback: return input (or best-guess crop)
                pred = crop_to_objects(inp)
            preds.append(pred)

        submission[tid] = preds
        if method == "failed":
            logger.debug(f"{tid}: NO MATCH")
        else:
            logger.debug(f"{tid}: {method}")

    # Summary
    solved = sum(v for k, v in stats.items() if k != "failed")
    total = len(challenges)
    logger.info(f"\nSolved: {solved}/{total}")
    logger.info("Method breakdown:")
    for method, count in sorted(stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {method:30s}: {count}")

    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    logger.info(f"\nSubmission written → {output_path}")
    return submission, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="RE-ARC auto-solver")
    parser.add_argument("--challenges", required=True, help="Path to challenge JSON")
    parser.add_argument("--output", required=True, help="Output submission JSON path")
    args = parser.parse_args()
    run_solver(args.challenges, args.output)


if __name__ == "__main__":
    main()
