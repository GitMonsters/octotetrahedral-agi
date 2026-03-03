#!/usr/bin/env python3
"""
ARC Program Synthesis Engine — General-purpose rule induction for ARC-AGI.

Instead of 80+ task-specific operations, uses ~25 general primitives +
systematic search to find transformation rules that work on ALL training pairs.

Key insight: Find the simplest program (sequence of primitives) that maps
input → output consistently across all training examples.
"""

import json
import glob
import os
import sys
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Any, Callable
from collections import Counter, defaultdict
from itertools import product


# ─── Grid utilities ──────────────────────────────────────────────────────────

Grid = List[List[int]]

def to_np(grid: Grid) -> np.ndarray:
    return np.array(grid, dtype=int)

def to_grid(arr: np.ndarray) -> Grid:
    return arr.tolist()

def grid_eq(a: Grid, b: Grid) -> bool:
    return to_np(a).shape == to_np(b).shape and np.array_equal(to_np(a), to_np(b))

def grid_shape(g: Grid) -> Tuple[int, int]:
    a = to_np(g)
    return a.shape[0], a.shape[1]

def background_color(g: Grid) -> int:
    """Most common color (typically 0 = black)."""
    a = to_np(g)
    counts = Counter(a.flatten())
    return counts.most_common(1)[0][0]

def colors_in(g: Grid) -> Set[int]:
    return set(to_np(g).flatten())

def color_count(g: Grid) -> Dict[int, int]:
    return dict(Counter(to_np(g).flatten()))


# ─── Object detection (connected components) ────────────────────────────────

class ArcObject:
    """A connected component in a grid."""
    def __init__(self, cells: List[Tuple[int, int]], color: int, grid_shape: Tuple[int, int]):
        self.cells = cells
        self.color = color
        self.grid_shape = grid_shape
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        self.min_r, self.max_r = min(rows), max(rows)
        self.min_c, self.max_c = min(cols), max(cols)
        self.height = self.max_r - self.min_r + 1
        self.width = self.max_c - self.min_c + 1
        self.size = len(cells)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.min_r, self.min_c, self.max_r, self.max_c)

    @property
    def shape_grid(self) -> np.ndarray:
        """Cropped grid of just this object (color on bg 0)."""
        g = np.zeros((self.height, self.width), dtype=int)
        for r, c in self.cells:
            g[r - self.min_r, c - self.min_c] = self.color
        return g

    @property
    def mask(self) -> np.ndarray:
        """Boolean mask of this object in full grid."""
        m = np.zeros(self.grid_shape, dtype=bool)
        for r, c in self.cells:
            m[r, c] = True
        return m


def find_objects(grid: Grid, bg: Optional[int] = None, diagonal: bool = False) -> List[ArcObject]:
    """Find connected components (objects) in the grid."""
    a = to_np(grid)
    h, w = a.shape
    if bg is None:
        bg = background_color(grid)
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    if diagonal:
        neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]

    for r in range(h):
        for c in range(w):
            if visited[r, c] or a[r, c] == bg:
                continue
            # BFS
            color = a[r, c]
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= h or cc < 0 or cc >= w:
                    continue
                if visited[cr, cc] or a[cr, cc] != color:
                    continue
                visited[cr, cc] = True
                cells.append((cr, cc))
                for dr, dc in neighbors:
                    stack.append((cr + dr, cc + dc))
            if cells:
                objects.append(ArcObject(cells, color, (h, w)))

    return sorted(objects, key=lambda o: o.size, reverse=True)


def find_objects_multicolor(grid: Grid, bg: Optional[int] = None) -> List[ArcObject]:
    """Find connected components treating all non-bg colors as one."""
    a = to_np(grid)
    h, w = a.shape
    if bg is None:
        bg = background_color(grid)
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or a[r, c] == bg:
                continue
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= h or cc < 0 or cc >= w:
                    continue
                if visited[cr, cc] or a[cr, cc] == bg:
                    continue
                visited[cr, cc] = True
                cells.append((cr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    stack.append((cr + dr, cc + dc))
            if cells:
                # Use dominant color
                colors = [a[r2,c2] for r2,c2 in cells]
                dom_color = Counter(colors).most_common(1)[0][0]
                objects.append(ArcObject(cells, dom_color, (h, w)))

    return sorted(objects, key=lambda o: o.size, reverse=True)


# ─── Primitive operations ────────────────────────────────────────────────────

def rotate90(g: Grid) -> Grid:
    return to_grid(np.rot90(to_np(g), k=-1))

def rotate180(g: Grid) -> Grid:
    return to_grid(np.rot90(to_np(g), k=2))

def rotate270(g: Grid) -> Grid:
    return to_grid(np.rot90(to_np(g), k=1))

def flip_h(g: Grid) -> Grid:
    return to_grid(np.fliplr(to_np(g)))

def flip_v(g: Grid) -> Grid:
    return to_grid(np.flipud(to_np(g)))

def transpose(g: Grid) -> Grid:
    return to_grid(to_np(g).T)

def crop_to_content(g: Grid, bg: Optional[int] = None) -> Grid:
    """Remove surrounding background rows/cols."""
    a = to_np(g)
    if bg is None:
        bg = background_color(g)
    mask = a != bg
    if not mask.any():
        return g
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return to_grid(a[rows][:, cols])

def scale_up(g: Grid, factor: int) -> Grid:
    """Scale grid by integer factor."""
    a = to_np(g)
    return to_grid(np.repeat(np.repeat(a, factor, axis=0), factor, axis=1))

def scale_down(g: Grid, factor: int) -> Grid:
    """Downscale grid by integer factor (top-left pixel of each block)."""
    a = to_np(g)
    h, w = a.shape
    if h % factor != 0 or w % factor != 0:
        return g
    return to_grid(a[::factor, ::factor])

def tile(g: Grid, nh: int, nw: int) -> Grid:
    """Tile grid nh times vertically, nw times horizontally."""
    a = to_np(g)
    return to_grid(np.tile(a, (nh, nw)))

def overlay(base: Grid, top: Grid, bg: int = 0) -> Grid:
    """Overlay top grid onto base, ignoring bg color in top."""
    a = to_np(base).copy()
    b = to_np(top)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    mask = b[:h, :w] != bg
    a[:h, :w][mask] = b[:h, :w][mask]
    return to_grid(a)

def gravity_down(g: Grid, bg: int = 0) -> Grid:
    """Drop non-bg cells down (gravity)."""
    a = to_np(g).copy()
    h, w = a.shape
    for c in range(w):
        col = a[:, c]
        non_bg = col[col != bg]
        a[:, c] = bg
        a[h-len(non_bg):, c] = non_bg
    return to_grid(a)

def gravity_up(g: Grid, bg: int = 0) -> Grid:
    a = to_np(g).copy()
    h, w = a.shape
    for c in range(w):
        col = a[:, c]
        non_bg = col[col != bg]
        a[:, c] = bg
        a[:len(non_bg), c] = non_bg
    return to_grid(a)

def gravity_left(g: Grid, bg: int = 0) -> Grid:
    a = to_np(g).copy()
    h, w = a.shape
    for r in range(h):
        row = a[r, :]
        non_bg = row[row != bg]
        a[r, :] = bg
        a[r, :len(non_bg)] = non_bg
    return to_grid(a)

def gravity_right(g: Grid, bg: int = 0) -> Grid:
    a = to_np(g).copy()
    h, w = a.shape
    for r in range(h):
        row = a[r, :]
        non_bg = row[row != bg]
        a[r, :] = bg
        a[r, w-len(non_bg):] = non_bg
    return to_grid(a)

def fill_enclosed(g: Grid, fill_color: int, bg: int = 0) -> Grid:
    """Fill enclosed regions (not reachable from border) with fill_color."""
    a = to_np(g).copy()
    h, w = a.shape
    reachable = np.zeros((h, w), dtype=bool)
    # BFS from all border bg cells
    stack = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and a[r, c] == bg:
                stack.append((r, c))
                reachable[r, c] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and a[nr, nc] == bg:
                reachable[nr, nc] = True
                stack.append((nr, nc))
    # Fill unreachable bg cells
    for r in range(h):
        for c in range(w):
            if a[r, c] == bg and not reachable[r, c]:
                a[r, c] = fill_color
    return to_grid(a)


# ─── Rule inference strategies ───────────────────────────────────────────────

class RuleCandidate:
    """A candidate transformation rule."""
    def __init__(self, name: str, transform_fn: Callable, priority: float = 0.0):
        self.name = name
        self.transform_fn = transform_fn
        self.priority = priority

    def apply(self, grid: Grid) -> Optional[Grid]:
        try:
            result = self.transform_fn(grid)
            if result is not None:
                a = to_np(result)
                if a.shape[0] > 0 and a.shape[1] > 0 and a.shape[0] <= 30 and a.shape[1] <= 30:
                    return result
        except Exception:
            pass
        return None


def infer_color_mapping(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[RuleCandidate]:
    """Check if output is input with colors remapped."""
    mappings = []
    for inp, out in train_pairs:
        a_in, a_out = to_np(inp), to_np(out)
        if a_in.shape != a_out.shape:
            return None
        mapping = {}
        valid = True
        for r in range(a_in.shape[0]):
            for c in range(a_in.shape[1]):
                ci, co = int(a_in[r,c]), int(a_out[r,c])
                if ci in mapping:
                    if mapping[ci] != co:
                        valid = False
                        break
                else:
                    mapping[ci] = co
            if not valid:
                break
        if not valid:
            return None
        mappings.append(mapping)

    # Check consistency across pairs
    if not mappings:
        return None
    final_map = mappings[0]
    for m in mappings[1:]:
        for k, v in m.items():
            if k in final_map and final_map[k] != v:
                return None
            final_map[k] = v

    def apply_map(grid):
        a = to_np(grid).copy()
        for old_c, new_c in final_map.items():
            a[to_np(grid) == old_c] = new_c
        return to_grid(a)

    return RuleCandidate(f"color_map({final_map})", apply_map, priority=10.0)


def infer_geometric_transform(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is a geometric transform of input."""
    transforms = [
        ("rotate90", rotate90),
        ("rotate180", rotate180),
        ("rotate270", rotate270),
        ("flip_h", flip_h),
        ("flip_v", flip_v),
        ("transpose", transpose),
    ]
    results = []
    for name, fn in transforms:
        if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
            results.append(RuleCandidate(name, fn, priority=9.0))
    return results


def infer_crop(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is a crop of the input."""
    rules = []

    # Crop to content (remove bg border)
    for bg_c in range(10):
        def make_crop(bg):
            return lambda g: crop_to_content(g, bg=bg)
        if all(grid_eq(crop_to_content(inp, bg=bg_c), out) for inp, out in train_pairs):
            rules.append(RuleCandidate(f"crop_content(bg={bg_c})", make_crop(bg_c), priority=8.0))

    # Crop to largest object
    for bg_c in range(10):
        def make_obj_crop(bg):
            def fn(g):
                objs = find_objects(g, bg=bg)
                if not objs:
                    return g
                obj = objs[0]  # largest
                return to_grid(obj.shape_grid)
            return fn
        try:
            if all(grid_eq(make_obj_crop(bg_c)(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(f"crop_largest_obj(bg={bg_c})", make_obj_crop(bg_c), priority=7.5))
        except Exception:
            pass

    return rules


def infer_scaling(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is a scaled version of input."""
    rules = []
    for factor in [2, 3, 4, 5]:
        if all(grid_eq(scale_up(inp, factor), out) for inp, out in train_pairs):
            def make_scale(f):
                return lambda g: scale_up(g, f)
            rules.append(RuleCandidate(f"scale_up({factor})", make_scale(factor), priority=8.0))

        if all(grid_eq(scale_down(inp, factor), out) for inp, out in train_pairs):
            def make_downscale(f):
                return lambda g: scale_down(g, f)
            rules.append(RuleCandidate(f"scale_down({factor})", make_downscale(factor), priority=8.0))

    return rules


def infer_tiling(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is a tiled version of input."""
    rules = []
    for nh in range(1, 6):
        for nw in range(1, 6):
            if nh == 1 and nw == 1:
                continue
            if all(grid_eq(tile(inp, nh, nw), out) for inp, out in train_pairs):
                def make_tile(h, w):
                    return lambda g: tile(g, h, w)
                rules.append(RuleCandidate(f"tile({nh},{nw})", make_tile(nh, nw), priority=7.0))
    return rules


def infer_gravity(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output has gravity applied."""
    rules = []
    for name, fn in [("gravity_down", gravity_down), ("gravity_up", gravity_up),
                      ("gravity_left", gravity_left), ("gravity_right", gravity_right)]:
        for bg in range(10):
            def make_grav(f, b):
                return lambda g: f(g, bg=b)
            if all(grid_eq(make_grav(fn, bg)(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(f"{name}(bg={bg})", make_grav(fn, bg), priority=7.0))
    return rules


def infer_fill(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output has enclosed regions filled."""
    rules = []
    for fill_c in range(1, 10):
        for bg in range(10):
            if fill_c == bg:
                continue
            def make_fill(fc, b):
                return lambda g: fill_enclosed(g, fill_color=fc, bg=b)
            if all(grid_eq(make_fill(fill_c, bg)(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(f"fill_enclosed(c={fill_c},bg={bg})", make_fill(fill_c, bg), priority=7.0))
    return rules


def infer_object_recolor(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if specific objects get recolored based on properties."""
    rules = []

    for bg in range(10):
        # Recolor smallest object to a specific color
        for target_color in range(10):
            if target_color == bg:
                continue

            def make_recolor_smallest(tc, b):
                def fn(g):
                    a = to_np(g).copy()
                    objs = find_objects(g, bg=b)
                    if not objs:
                        return g
                    smallest = min(objs, key=lambda o: o.size)
                    for r, c in smallest.cells:
                        a[r, c] = tc
                    return to_grid(a)
                return fn

            try:
                if all(grid_eq(make_recolor_smallest(target_color, bg)(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(
                        f"recolor_smallest(c={target_color},bg={bg})",
                        make_recolor_smallest(target_color, bg), priority=6.0))
            except Exception:
                pass

        # Recolor largest object
        for target_color in range(10):
            if target_color == bg:
                continue

            def make_recolor_largest(tc, b):
                def fn(g):
                    a = to_np(g).copy()
                    objs = find_objects(g, bg=b)
                    if not objs:
                        return g
                    largest = max(objs, key=lambda o: o.size)
                    for r, c in largest.cells:
                        a[r, c] = tc
                    return to_grid(a)
                return fn

            try:
                if all(grid_eq(make_recolor_largest(target_color, bg)(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(
                        f"recolor_largest(c={target_color},bg={bg})",
                        make_recolor_largest(target_color, bg), priority=6.0))
            except Exception:
                pass

    return rules


def infer_pattern_repeat(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is a repeating pattern from input subgrid."""
    rules = []
    for inp, out in train_pairs[:1]:  # Check first pair to get hypothesis
        a_out = to_np(out)
        oh, ow = a_out.shape
        # Try small subgrid sizes
        for ph in range(1, min(oh+1, 8)):
            for pw in range(1, min(ow+1, 8)):
                if oh % ph != 0 or ow % pw != 0:
                    continue
                pattern = a_out[:ph, :pw]
                tiled = np.tile(pattern, (oh // ph, ow // pw))
                if np.array_equal(tiled, a_out):
                    # Found a repeating pattern — does it come from input?
                    a_in = to_np(inp)
                    if a_in.shape == (ph, pw) and np.array_equal(a_in, pattern):
                        nh, nw = oh // ph, ow // pw
                        def make_tile(h, w):
                            return lambda g: tile(g, h, w)
                        # Verify on all pairs
                        if all(grid_eq(tile(i, nh, nw), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"tile({nh},{nw})", make_tile(nh, nw), priority=7.0))
    return rules


def infer_border_ops(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output adds/removes a border."""
    rules = []

    # Remove 1-pixel border
    def remove_border(g):
        a = to_np(g)
        if a.shape[0] < 3 or a.shape[1] < 3:
            return g
        return to_grid(a[1:-1, 1:-1])

    if all(grid_eq(remove_border(inp), out) for inp, out in train_pairs):
        rules.append(RuleCandidate("remove_border", remove_border, priority=7.0))

    # Add border with specific color
    for border_c in range(10):
        def make_add_border(bc):
            def fn(g):
                a = to_np(g)
                h, w = a.shape
                result = np.full((h+2, w+2), bc, dtype=int)
                result[1:-1, 1:-1] = a
                return to_grid(result)
            return fn
        if all(grid_eq(make_add_border(border_c)(inp), out) for inp, out in train_pairs):
            rules.append(RuleCandidate(f"add_border(c={border_c})", make_add_border(border_c), priority=7.0))

    return rules


def infer_per_cell_rule(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Infer rules based on local cell neighborhoods."""
    rules = []

    # Check: output cell = f(neighborhood of input cell)
    # Most common: fill based on neighbor count/colors

    # Simple: output same as input except bg cells adjacent to non-bg get colored
    for bg in range(10):
        for fill_c in range(10):
            if fill_c == bg:
                continue
            def make_adj_fill(b, fc):
                def fn(g):
                    a = to_np(g).copy()
                    h, w = a.shape
                    result = a.copy()
                    for r in range(h):
                        for c in range(w):
                            if a[r, c] == b:
                                # Check if any neighbor is non-bg
                                has_neighbor = False
                                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                    nr, nc = r+dr, c+dc
                                    if 0 <= nr < h and 0 <= nc < w and a[nr, nc] != b:
                                        has_neighbor = True
                                        break
                                if has_neighbor:
                                    result[r, c] = fc
                    return to_grid(result)
                return fn

            try:
                if all(grid_eq(make_adj_fill(bg, fill_c)(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(
                        f"adj_fill(bg={bg},c={fill_c})",
                        make_adj_fill(bg, fill_c), priority=5.0))
            except Exception:
                pass

    return rules


def infer_majority_vote(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is element-wise majority across multiple input regions."""
    rules = []

    # Check if input can be split into equal-sized subgrids
    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        ih, iw = a_in.shape
        oh, ow = a_out.shape

        # Try horizontal splits
        for n in [2, 3, 4]:
            if ih % n == 0:
                sh = ih // n
                if sh == oh and iw == ow:
                    regions = [a_in[i*sh:(i+1)*sh, :] for i in range(n)]
                    # Majority vote
                    result = np.zeros((sh, iw), dtype=int)
                    for r in range(sh):
                        for c in range(iw):
                            vals = [reg[r, c] for reg in regions]
                            result[r, c] = Counter(vals).most_common(1)[0][0]
                    if np.array_equal(result, a_out):
                        def make_hmaj(nn):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                s = h // nn
                                regs = [a[i*s:(i+1)*s, :] for i in range(nn)]
                                res = np.zeros((s, w), dtype=int)
                                for rr in range(s):
                                    for cc in range(w):
                                        vs = [reg[rr, cc] for reg in regs]
                                        res[rr, cc] = Counter(vs).most_common(1)[0][0]
                                return to_grid(res)
                            return fn
                        if all(grid_eq(make_hmaj(n)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"h_majority({n})", make_hmaj(n), priority=6.0))

        # Try vertical splits
        for n in [2, 3, 4]:
            if iw % n == 0:
                sw = iw // n
                if ih == oh and sw == ow:
                    regions = [a_in[:, i*sw:(i+1)*sw] for i in range(n)]
                    result = np.zeros((ih, sw), dtype=int)
                    for r in range(ih):
                        for c in range(sw):
                            vals = [reg[r, c] for reg in regions]
                            result[r, c] = Counter(vals).most_common(1)[0][0]
                    if np.array_equal(result, a_out):
                        def make_vmaj(nn):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                s = w // nn
                                regs = [a[:, i*s:(i+1)*s] for i in range(nn)]
                                res = np.zeros((h, s), dtype=int)
                                for rr in range(h):
                                    for cc in range(s):
                                        vs = [reg[rr, cc] for reg in regs]
                                        res[rr, cc] = Counter(vs).most_common(1)[0][0]
                                return to_grid(res)
                            return fn
                        if all(grid_eq(make_vmaj(n)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"v_majority({n})", make_vmaj(n), priority=6.0))

    return rules


def infer_xor_overlay(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is XOR/diff between input subgrids."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        ih, iw = a_in.shape
        oh, ow = a_out.shape

        # Split horizontally into 2 halves
        if ih % 2 == 0:
            sh = ih // 2
            if sh == oh and iw == ow:
                top, bot = a_in[:sh, :], a_in[sh:, :]

                # XOR-like: cells that differ
                for bg in range(10):
                    diff = np.where(top != bot, top, bg)
                    if np.array_equal(diff, a_out):
                        def make_hdiff(b):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                s = h // 2
                                t, bo = a[:s, :], a[s:, :]
                                return to_grid(np.where(t != bo, t, b))
                            return fn
                        if all(grid_eq(make_hdiff(bg)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"h_xor(bg={bg})", make_hdiff(bg), priority=6.0))

                    diff2 = np.where(top != bot, bot, bg)
                    if np.array_equal(diff2, a_out):
                        def make_hdiff2(b):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                s = h // 2
                                t, bo = a[:s, :], a[s:, :]
                                return to_grid(np.where(t != bo, bo, b))
                            return fn
                        if all(grid_eq(make_hdiff2(bg)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"h_xor_bot(bg={bg})", make_hdiff2(bg), priority=6.0))

        # Split vertically into 2 halves
        if iw % 2 == 0:
            sw = iw // 2
            if ih == oh and sw == ow:
                left, right = a_in[:, :sw], a_in[:, sw:]

                for bg in range(10):
                    diff = np.where(left != right, left, bg)
                    if np.array_equal(diff, a_out):
                        def make_vdiff(b):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                s = w // 2
                                l, r = a[:, :s], a[:, s:]
                                return to_grid(np.where(l != r, l, b))
                            return fn
                        if all(grid_eq(make_vdiff(bg)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"v_xor(bg={bg})", make_vdiff(bg), priority=6.0))

    return rules


def infer_subgrid_extraction(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is a subgrid extracted from input."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        ih, iw = a_in.shape
        oh, ow = a_out.shape

        if oh > ih or ow > iw:
            continue

        # Try all possible subgrid positions
        for r in range(ih - oh + 1):
            for c in range(iw - ow + 1):
                if np.array_equal(a_in[r:r+oh, c:c+ow], a_out):
                    # Fixed position extraction
                    def make_extract(rr, cc, hh, ww):
                        return lambda g: to_grid(to_np(g)[rr:rr+hh, cc:cc+ww])

                    if all(grid_eq(make_extract(r, c, oh, ow)(i), o) for i, o in train_pairs):
                        rules.append(RuleCandidate(
                            f"extract({r},{c},{oh},{ow})",
                            make_extract(r, c, oh, ow), priority=7.0))

    return rules


def infer_unique_subgrid(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is the unique/different subgrid among repeated patterns."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        ih, iw = a_in.shape
        oh, ow = a_out.shape

        # Try dividing input into oh x ow blocks
        if ih % oh == 0 and iw % ow == 0:
            nh, nw = ih // oh, iw // ow
            if nh * nw < 2:
                continue

            blocks = []
            for bi in range(nh):
                for bj in range(nw):
                    block = a_in[bi*oh:(bi+1)*oh, bj*ow:(bj+1)*ow]
                    blocks.append((bi, bj, block))

            # Find the unique block (one that differs from others)
            for idx, (bi, bj, block) in enumerate(blocks):
                others = [b for i, (_, _, b) in enumerate(blocks) if i != idx]
                if all(np.array_equal(others[0], o) for o in others[1:]):
                    # This block is the unique one
                    if np.array_equal(block, a_out):
                        def make_unique_block(bh, bw, nh_, nw_):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                blks = []
                                for i in range(nh_):
                                    for j in range(nw_):
                                        blks.append(a[i*bh:(i+1)*bh, j*bw:(j+1)*bw])
                                # Find unique
                                for i, b in enumerate(blks):
                                    others = [bl for j, bl in enumerate(blks) if j != i]
                                    if all(not np.array_equal(b, o) for o in others):
                                        return to_grid(b)
                                return to_grid(blks[0])
                            return fn
                        if all(grid_eq(make_unique_block(oh, ow, nh, nw)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(
                                f"unique_block({oh}x{ow})",
                                make_unique_block(oh, ow, nh, nw), priority=7.5))

    return rules


def infer_color_scale(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if each input cell maps to an NxN block based on its color."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        ih, iw = a_in.shape
        oh, ow = a_out.shape

        if oh % ih != 0 or ow % iw != 0:
            continue
        bh, bw = oh // ih, ow // iw
        if bh < 2 or bw < 2 or bh > 10 or bw > 10:
            continue

        # For each color, extract the block pattern
        color_blocks = {}
        consistent = True
        for r in range(ih):
            for c in range(iw):
                color = int(a_in[r, c])
                block = a_out[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                if color in color_blocks:
                    if not np.array_equal(color_blocks[color], block):
                        consistent = False
                        break
                else:
                    color_blocks[color] = block.copy()
            if not consistent:
                break

        if consistent and color_blocks:
            def make_color_scale(blocks, bh_, bw_):
                def fn(g):
                    a = to_np(g)
                    h, w = a.shape
                    result = np.zeros((h * bh_, w * bw_), dtype=int)
                    for r in range(h):
                        for c in range(w):
                            color = int(a[r, c])
                            if color in blocks:
                                result[r*bh_:(r+1)*bh_, c*bw_:(c+1)*bw_] = blocks[color]
                            # else leave as 0
                    return to_grid(result)
                return fn

            if all(grid_eq(make_color_scale(color_blocks, bh, bw)(i), o) for i, o in train_pairs):
                rules.append(RuleCandidate(
                    f"color_scale({bh}x{bw})",
                    make_color_scale(color_blocks, bh, bw), priority=8.5))

    return rules


def infer_symmetry_completion(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output completes a symmetric pattern from input."""
    rules = []

    # Check horizontal symmetry completion
    def complete_h_sym(g, bg=0):
        a = to_np(g).copy()
        h, w = a.shape
        for r in range(h):
            for c in range(w):
                mirror_c = w - 1 - c
                if a[r, c] == bg and a[r, mirror_c] != bg:
                    a[r, c] = a[r, mirror_c]
                elif a[r, mirror_c] == bg and a[r, c] != bg:
                    a[r, mirror_c] = a[r, c]
        return to_grid(a)

    def complete_v_sym(g, bg=0):
        a = to_np(g).copy()
        h, w = a.shape
        for r in range(h):
            mirror_r = h - 1 - r
            for c in range(w):
                if a[r, c] == bg and a[mirror_r, c] != bg:
                    a[r, c] = a[mirror_r, c]
                elif a[mirror_r, c] == bg and a[r, c] != bg:
                    a[mirror_r, c] = a[r, c]
        return to_grid(a)

    def complete_diag_sym(g, bg=0):
        a = to_np(g).copy()
        h, w = a.shape
        if h != w:
            return g
        for r in range(h):
            for c in range(w):
                if a[r, c] == bg and a[c, r] != bg:
                    a[r, c] = a[c, r]
                elif a[c, r] == bg and a[r, c] != bg:
                    a[c, r] = a[r, c]
        return to_grid(a)

    for bg in range(10):
        for name, fn in [("h_sym", complete_h_sym), ("v_sym", complete_v_sym), ("diag_sym", complete_diag_sym)]:
            def make_sym(f, b):
                return lambda g: f(g, bg=b)
            try:
                if all(grid_eq(make_sym(fn, bg)(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(f"{name}_complete(bg={bg})", make_sym(fn, bg), priority=7.0))
            except Exception:
                pass

    return rules


def infer_line_extension(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output extends lines from colored cells."""
    rules = []

    for bg in [0]:
        # Extend each non-bg cell in all 4 cardinal directions until hitting non-bg
        for direction in ['right', 'left', 'up', 'down']:
            def make_extend(d, b):
                def fn(g):
                    a = to_np(g).copy()
                    h, w = a.shape
                    for r in range(h):
                        for c in range(w):
                            if a[r, c] != b:
                                color = a[r, c]
                                if d == 'right':
                                    for cc in range(c+1, w):
                                        if to_np(g)[cc if d in ['down','up'] else r, cc if d in ['right','left'] else c] != b:
                                            break
                                        a[r, cc] = color
                                elif d == 'left':
                                    for cc in range(c-1, -1, -1):
                                        if to_np(g)[r, cc] != b:
                                            break
                                        a[r, cc] = color
                                elif d == 'down':
                                    for rr in range(r+1, h):
                                        if to_np(g)[rr, c] != b:
                                            break
                                        a[rr, c] = color
                                elif d == 'up':
                                    for rr in range(r-1, -1, -1):
                                        if to_np(g)[rr, c] != b:
                                            break
                                        a[rr, c] = color
                    return to_grid(a)
                return fn

            try:
                if all(grid_eq(make_extend(direction, bg)(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(
                        f"extend_{direction}(bg={bg})",
                        make_extend(direction, bg), priority=6.5))
            except Exception:
                pass

        # Extend all directions at once
        def make_extend_all(b):
            def fn(g):
                a = to_np(g)
                h, w = a.shape
                result = a.copy()
                orig = a.copy()
                for r in range(h):
                    for c in range(w):
                        if orig[r, c] != b:
                            color = int(orig[r, c])
                            # Right
                            for cc in range(c+1, w):
                                if orig[r, cc] != b: break
                                result[r, cc] = color
                            # Left
                            for cc in range(c-1, -1, -1):
                                if orig[r, cc] != b: break
                                result[r, cc] = color
                            # Down
                            for rr in range(r+1, h):
                                if orig[rr, c] != b: break
                                result[rr, c] = color
                            # Up
                            for rr in range(r-1, -1, -1):
                                if orig[rr, c] != b: break
                                result[rr, c] = color
                return to_grid(result)
            return fn

        try:
            if all(grid_eq(make_extend_all(bg)(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(f"extend_all(bg={bg})", make_extend_all(bg), priority=6.5))
        except Exception:
            pass

    return rules


def infer_object_stamp(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if a pattern is stamped at positions marked by specific pixels."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        if a_in.shape != a_out.shape:
            continue

        bg = background_color(inp)
        objs = find_objects(inp, bg=bg)
        if len(objs) < 2:
            continue

        # Check if all objects except the largest share the same shape
        if len(objs) >= 2:
            largest = objs[0]
            pattern = largest.shape_grid
            ph, pw = pattern.shape

            # Check if pattern is stamped at positions of single-cell objects
            single_cells = [o for o in objs if o.size == 1]
            if len(single_cells) >= 1:
                def make_stamp(pat, b):
                    def fn(g):
                        a = to_np(g).copy()
                        h, w = a.shape
                        bg_ = background_color(g)
                        objects = find_objects(g, bg=bg_)
                        if not objects:
                            return g
                        largest_ = objects[0]
                        pattern_ = largest_.shape_grid
                        p_h, p_w = pattern_.shape
                        singles = [o for o in objects if o.size == 1]
                        for s in singles:
                            sr, sc = s.cells[0]
                            for pr in range(p_h):
                                for pc in range(p_w):
                                    if pattern_[pr, pc] != 0:
                                        nr = sr - p_h//2 + pr
                                        nc = sc - p_w//2 + pc
                                        if 0 <= nr < h and 0 <= nc < w:
                                            a[nr, nc] = int(pattern_[pr, pc])
                        return to_grid(a)
                    return fn

                try:
                    if all(grid_eq(make_stamp(pattern, bg)(i), o) for i, o in train_pairs):
                        rules.append(RuleCandidate("object_stamp", make_stamp(pattern, bg), priority=6.0))
                except Exception:
                    pass

    return rules


def infer_mirror_halves(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output mirrors one half onto the other."""
    rules = []

    # Mirror left half to right
    def mirror_lr(g):
        a = to_np(g).copy()
        h, w = a.shape
        for r in range(h):
            for c in range(w // 2):
                a[r, w - 1 - c] = a[r, c]
        return to_grid(a)

    # Mirror right half to left
    def mirror_rl(g):
        a = to_np(g).copy()
        h, w = a.shape
        for r in range(h):
            for c in range(w // 2, w):
                a[r, w - 1 - c] = a[r, c]
        return to_grid(a)

    # Mirror top half to bottom
    def mirror_tb(g):
        a = to_np(g).copy()
        h, w = a.shape
        for r in range(h // 2):
            a[h - 1 - r, :] = a[r, :]
        return to_grid(a)

    # Mirror bottom half to top
    def mirror_bt(g):
        a = to_np(g).copy()
        h, w = a.shape
        for r in range(h // 2, h):
            a[h - 1 - r, :] = a[r, :]
        return to_grid(a)

    for name, fn in [("mirror_lr", mirror_lr), ("mirror_rl", mirror_rl),
                      ("mirror_tb", mirror_tb), ("mirror_bt", mirror_bt)]:
        try:
            if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(name, fn, priority=7.0))
        except Exception:
            pass

    return rules


def infer_remove_color(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output removes cells of a specific color (replace with bg)."""
    rules = []

    for bg in range(10):
        for remove_c in range(10):
            if remove_c == bg:
                continue
            def make_remove(rc, b):
                def fn(g):
                    a = to_np(g).copy()
                    a[a == rc] = b
                    return to_grid(a)
                return fn

            try:
                if all(grid_eq(make_remove(remove_c, bg)(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(
                        f"remove_color({remove_c},bg={bg})",
                        make_remove(remove_c, bg), priority=7.5))
            except Exception:
                pass

    return rules


def infer_keep_color(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output keeps only one color (set all others to bg)."""
    rules = []

    for bg in range(10):
        for keep_c in range(10):
            if keep_c == bg:
                continue
            def make_keep(kc, b):
                def fn(g):
                    a = to_np(g).copy()
                    mask = (a != kc) & (a != b)
                    a[mask] = b
                    return to_grid(a)
                return fn

            try:
                if all(grid_eq(make_keep(keep_c, bg)(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(
                        f"keep_color({keep_c},bg={bg})",
                        make_keep(keep_c, bg), priority=7.5))
            except Exception:
                pass

    return rules


def infer_row_col_ops(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Infer row/column based operations."""
    rules = []

    # Sort rows by some criterion
    for bg in [0]:
        # Sort rows by number of non-bg cells
        def sort_rows_asc(g, b=bg):
            a = to_np(g)
            counts = [(np.sum(a[r] != b), r) for r in range(a.shape[0])]
            counts.sort()
            return to_grid(a[[r for _, r in counts]])

        def sort_rows_desc(g, b=bg):
            a = to_np(g)
            counts = [(np.sum(a[r] != b), r) for r in range(a.shape[0])]
            counts.sort(reverse=True)
            return to_grid(a[[r for _, r in counts]])

        for name, fn in [("sort_rows_asc", sort_rows_asc), ("sort_rows_desc", sort_rows_desc)]:
            try:
                if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(name, fn, priority=6.5))
            except Exception:
                pass

    # Deduplicate rows
    def dedup_rows(g):
        a = to_np(g)
        seen = []
        result = []
        for r in range(a.shape[0]):
            row = tuple(a[r])
            if row not in seen:
                seen.append(row)
                result.append(a[r])
        return to_grid(np.array(result))

    def dedup_cols(g):
        a = to_np(g).T
        seen = []
        result = []
        for r in range(a.shape[0]):
            row = tuple(a[r])
            if row not in seen:
                seen.append(row)
                result.append(a[r])
        return to_grid(np.array(result).T)

    for name, fn in [("dedup_rows", dedup_rows), ("dedup_cols", dedup_cols)]:
        try:
            if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(name, fn, priority=6.5))
        except Exception:
            pass

    return rules


def infer_split_recombine(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output combines input subgrids using overlay/OR/AND."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        ih, iw = a_in.shape
        oh, ow = a_out.shape

        # Try horizontal split + overlay
        if ih % 2 == 0:
            sh = ih // 2
            if sh == oh and iw == ow:
                top, bot = a_in[:sh, :], a_in[sh:, :]

                for bg in [0]:
                    # OR overlay: non-bg from either
                    result = np.where(top != bg, top, bot)
                    if np.array_equal(result, a_out):
                        def make_hor(b):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                s = h // 2
                                t, bo = a[:s, :], a[s:, :]
                                return to_grid(np.where(t != b, t, bo))
                            return fn
                        if all(grid_eq(make_hor(bg)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"h_or_overlay(bg={bg})", make_hor(bg), priority=6.5))

                    # AND: keep only where both are non-bg
                    result2 = np.where((top != bg) & (bot != bg), top, bg)
                    if np.array_equal(result2, a_out):
                        def make_hand(b):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                s = h // 2
                                t, bo = a[:s, :], a[s:, :]
                                return to_grid(np.where((t != b) & (bo != b), t, b))
                            return fn
                        if all(grid_eq(make_hand(bg)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"h_and_overlay(bg={bg})", make_hand(bg), priority=6.5))

        # Try vertical split + overlay
        if iw % 2 == 0:
            sw = iw // 2
            if ih == oh and sw == ow:
                left, right = a_in[:, :sw], a_in[:, sw:]

                for bg in [0]:
                    result = np.where(left != bg, left, right)
                    if np.array_equal(result, a_out):
                        def make_ver(b):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                s = w // 2
                                l, r = a[:, :s], a[:, s:]
                                return to_grid(np.where(l != b, l, r))
                            return fn
                        if all(grid_eq(make_ver(bg)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"v_or_overlay(bg={bg})", make_ver(bg), priority=6.5))

                    result2 = np.where((left != bg) & (right != bg), left, bg)
                    if np.array_equal(result2, a_out):
                        def make_vand(b):
                            def fn(g):
                                a = to_np(g)
                                h, w = a.shape
                                s = w // 2
                                l, r = a[:, :s], a[:, s:]
                                return to_grid(np.where((l != b) & (r != b), l, b))
                            return fn
                        if all(grid_eq(make_vand(bg)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(f"v_and_overlay(bg={bg})", make_vand(bg), priority=6.5))

    return rules


def infer_flood_from_markers(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output flood-fills from marker cells."""
    rules = []

    for marker_c in range(1, 10):
        for bg in [0]:
            def make_flood(mc, b):
                def fn(g):
                    a = to_np(g).copy()
                    h, w = a.shape
                    # Find marker cells
                    markers = list(zip(*np.where(a == mc)))
                    if not markers:
                        return g
                    # Flood fill from each marker
                    for mr, mc_ in markers:
                        stack = [(mr, mc_)]
                        visited = set()
                        while stack:
                            r, c = stack.pop()
                            if (r, c) in visited:
                                continue
                            if r < 0 or r >= h or c < 0 or c >= w:
                                continue
                            if a[r, c] != b and a[r, c] != mc:
                                continue
                            visited.add((r, c))
                            a[r, c] = mc
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                stack.append((r+dr, c+dc))
                    return to_grid(a)
                return fn

            try:
                if all(grid_eq(make_flood(marker_c, bg)(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(
                        f"flood_from_markers(c={marker_c},bg={bg})",
                        make_flood(marker_c, bg), priority=6.0))
            except Exception:
                pass

    return rules


def infer_replace_pattern(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output replaces a small pattern with another pattern."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        if a_in.shape != a_out.shape:
            continue
        h, w = a_in.shape

        # Find diff positions
        diff = a_in != a_out
        if not diff.any():
            continue

        diff_rows = np.any(diff, axis=1)
        diff_cols = np.any(diff, axis=0)
        r_min, r_max = np.where(diff_rows)[0][[0, -1]]
        c_min, c_max = np.where(diff_cols)[0][[0, -1]]

        ph = r_max - r_min + 1
        pw = c_max - c_min + 1

        if ph > 5 or pw > 5:
            continue

        # Extract old and new patterns
        old_pat = a_in[r_min:r_max+1, c_min:c_max+1].copy()
        new_pat = a_out[r_min:r_max+1, c_min:c_max+1].copy()

        # Check if this is a consistent replacement across all pairs
        def make_replace(op, np_):
            def fn(g):
                a = to_np(g).copy()
                h, w = a.shape
                for r in range(h - op.shape[0] + 1):
                    for c in range(w - op.shape[1] + 1):
                        if np.array_equal(a[r:r+op.shape[0], c:c+op.shape[1]], op):
                            a[r:r+np_.shape[0], c:c+np_.shape[1]] = np_
                return to_grid(a)
            return fn

        try:
            if all(grid_eq(make_replace(old_pat, new_pat)(i), o) for i, o in train_pairs):
                rules.append(RuleCandidate(
                    f"replace_pattern({ph}x{pw})",
                    make_replace(old_pat, new_pat), priority=7.0))
        except Exception:
            pass

    return rules


def infer_wallpaper_tiling(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if output is a wallpaper-style tiling with alternating flips."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        ih, iw = a_in.shape
        oh, ow = a_out.shape

        if oh % ih != 0 or ow % iw != 0:
            continue
        nh, nw = oh // ih, ow // iw
        if nh < 2 or nw < 2:
            continue

        # Determine the transform for each block
        transforms = {
            'orig': lambda x: x,
            'flip_h': lambda x: np.fliplr(x),
            'flip_v': lambda x: np.flipud(x),
            'flip_hv': lambda x: np.flipud(np.fliplr(x)),
            'rot90': lambda x: np.rot90(x, 1),
            'rot180': lambda x: np.rot90(x, 2),
            'rot270': lambda x: np.rot90(x, -1),
            'transpose': lambda x: x.T,
        }

        block_pattern = []
        for br in range(nh):
            row_pattern = []
            for bc in range(nw):
                block = a_out[br*ih:(br+1)*ih, bc*iw:(bc+1)*iw]
                found = None
                for tname, tfn in transforms.items():
                    try:
                        if np.array_equal(tfn(a_in), block):
                            found = tname
                            break
                    except Exception:
                        pass
                if found is None:
                    break
                row_pattern.append(found)
            if len(row_pattern) != nw:
                break
            block_pattern.append(row_pattern)

        if len(block_pattern) != nh:
            continue

        # Check if row pattern alternates (most common ARC pattern)
        # Check if each row has same transform
        row_transforms = [block_pattern[r][0] for r in range(nh)]
        all_same_in_row = all(
            all(block_pattern[r][c] == block_pattern[r][0] for c in range(nw))
            for r in range(nh)
        )

        if all_same_in_row:
            def make_wallpaper(nh_, nw_, row_ts):
                transform_map = {
                    'orig': lambda x: x,
                    'flip_h': lambda x: np.fliplr(x),
                    'flip_v': lambda x: np.flipud(x),
                    'flip_hv': lambda x: np.flipud(np.fliplr(x)),
                    'rot90': lambda x: np.rot90(x, 1),
                    'rot180': lambda x: np.rot90(x, 2),
                    'rot270': lambda x: np.rot90(x, -1),
                    'transpose': lambda x: x.T,
                }
                def fn(g):
                    a = to_np(g)
                    h, w = a.shape
                    result = np.zeros((h * nh_, w * nw_), dtype=int)
                    for br in range(nh_):
                        tfn = transform_map[row_ts[br]]
                        block = tfn(a)
                        for bc in range(nw_):
                            result[br*h:(br+1)*h, bc*w:(bc+1)*w] = block
                    return to_grid(result)
                return fn

            if all(grid_eq(make_wallpaper(nh, nw, row_transforms)(i), o) for i, o in train_pairs):
                rules.append(RuleCandidate(
                    f"wallpaper({nh}x{nw})",
                    make_wallpaper(nh, nw, row_transforms), priority=8.0))

    return rules


def infer_grid_segmentation(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check if input is segmented by lines and regions are processed."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        h, w = a_in.shape

        # Find horizontal divider lines (full rows of single color)
        for divider_c in range(10):
            h_dividers = []
            for r in range(h):
                if np.all(a_in[r, :] == divider_c):
                    h_dividers.append(r)

            v_dividers = []
            for c in range(w):
                if np.all(a_in[:, c] == divider_c):
                    v_dividers.append(c)

            if not h_dividers and not v_dividers:
                continue

            # Extract regions between dividers
            h_boundaries = [-1] + h_dividers + [h]
            v_boundaries = [-1] + v_dividers + [w]

            regions = []
            for i in range(len(h_boundaries) - 1):
                for j in range(len(v_boundaries) - 1):
                    r1 = h_boundaries[i] + 1
                    r2 = h_boundaries[i + 1]
                    c1 = v_boundaries[j] + 1
                    c2 = v_boundaries[j + 1]
                    if r1 < r2 and c1 < c2:
                        regions.append(a_in[r1:r2, c1:c2])

            if not regions:
                continue

            # Check if output is one of the regions
            for reg in regions:
                if np.array_equal(reg, a_out):
                    # Which region? Check if consistently the same criterion
                    break

            # Check if output is OR overlay of all regions
            if regions and all(r.shape == regions[0].shape for r in regions):
                shape = regions[0].shape
                if a_out.shape == shape:
                    # OR overlay
                    bg = 0
                    result = np.full(shape, bg, dtype=int)
                    for reg in regions:
                        mask = reg != bg
                        result[mask] = reg[mask]
                    if np.array_equal(result, a_out):
                        def make_seg_or(dc, b):
                            def fn(g):
                                a = to_np(g)
                                hh, ww = a.shape
                                hd = [r for r in range(hh) if np.all(a[r, :] == dc)]
                                vd = [c for c in range(ww) if np.all(a[:, c] == dc)]
                                hb = [-1] + hd + [hh]
                                vb = [-1] + vd + [ww]
                                regs = []
                                for i in range(len(hb)-1):
                                    for j in range(len(vb)-1):
                                        r1, r2 = hb[i]+1, hb[i+1]
                                        c1, c2 = vb[j]+1, vb[j+1]
                                        if r1 < r2 and c1 < c2:
                                            regs.append(a[r1:r2, c1:c2])
                                if not regs:
                                    return g
                                s = regs[0].shape
                                res = np.full(s, b, dtype=int)
                                for reg in regs:
                                    if reg.shape == s:
                                        mask = reg != b
                                        res[mask] = reg[mask]
                                return to_grid(res)
                            return fn
                        if all(grid_eq(make_seg_or(divider_c, 0)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(
                                f"seg_or_overlay(div={divider_c})",
                                make_seg_or(divider_c, 0), priority=7.0))

                    # AND overlay
                    result2 = np.full(shape, bg, dtype=int)
                    for r in range(shape[0]):
                        for c in range(shape[1]):
                            vals = [reg[r, c] for reg in regions if reg.shape == shape]
                            non_bg = [v for v in vals if v != bg]
                            if len(non_bg) == len(vals):  # all non-bg
                                result2[r, c] = non_bg[0]
                    if np.array_equal(result2, a_out):
                        def make_seg_and(dc, b):
                            def fn(g):
                                a = to_np(g)
                                hh, ww = a.shape
                                hd = [r for r in range(hh) if np.all(a[r, :] == dc)]
                                vd = [c for c in range(ww) if np.all(a[:, c] == dc)]
                                hb = [-1] + hd + [hh]
                                vb = [-1] + vd + [ww]
                                regs = []
                                for i in range(len(hb)-1):
                                    for j in range(len(vb)-1):
                                        r1, r2 = hb[i]+1, hb[i+1]
                                        c1, c2 = vb[j]+1, vb[j+1]
                                        if r1 < r2 and c1 < c2:
                                            regs.append(a[r1:r2, c1:c2])
                                if not regs:
                                    return g
                                s = regs[0].shape
                                res = np.full(s, b, dtype=int)
                                for rr in range(s[0]):
                                    for cc in range(s[1]):
                                        vs = [reg[rr, cc] for reg in regs if reg.shape == s]
                                        nb = [v for v in vs if v != b]
                                        if len(nb) == len(vs):
                                            res[rr, cc] = nb[0]
                                return to_grid(res)
                            return fn
                        if all(grid_eq(make_seg_and(divider_c, 0)(i), o) for i, o in train_pairs):
                            rules.append(RuleCandidate(
                                f"seg_and_overlay(div={divider_c})",
                                make_seg_and(divider_c, 0), priority=7.0))

    return rules


def infer_diagonal_flip(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Check diagonal symmetry operations."""
    rules = []

    def flip_main_diag(g):
        """Flip along main diagonal (transpose)."""
        return to_grid(to_np(g).T)

    def flip_anti_diag(g):
        """Flip along anti-diagonal."""
        a = to_np(g)
        return to_grid(np.fliplr(np.flipud(a.T)))

    for name, fn in [("flip_main_diag", flip_main_diag), ("flip_anti_diag", flip_anti_diag)]:
        try:
            if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(name, fn, priority=8.0))
        except Exception:
            pass

    return rules


def infer_boolean_ops(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Boolean operations between different colored layers."""
    rules = []

    for inp, out in train_pairs[:1]:
        a_in, a_out = to_np(inp), to_np(out)
        if a_in.shape != a_out.shape:
            continue

        bg = 0
        in_colors = sorted(set(a_in.flatten()) - {bg})
        out_colors = sorted(set(a_out.flatten()) - {bg})

        if len(in_colors) != 2 or len(out_colors) != 1:
            continue

        c1, c2 = in_colors
        oc = out_colors[0]

        mask1 = a_in == c1
        mask2 = a_in == c2

        # OR: union of both colors
        or_result = np.where(mask1 | mask2, oc, bg)
        if np.array_equal(or_result, a_out):
            def make_bool_or(cc1, cc2, ooc, b):
                def fn(g):
                    a = to_np(g)
                    m1 = a == cc1
                    m2 = a == cc2
                    return to_grid(np.where(m1 | m2, ooc, b))
                return fn
            if all(grid_eq(make_bool_or(c1, c2, oc, bg)(i), o) for i, o in train_pairs):
                rules.append(RuleCandidate(f"bool_or({c1},{c2})→{oc}", make_bool_or(c1, c2, oc, bg), priority=7.0))

        # AND: intersection
        and_result = np.where(mask1 & mask2, oc, bg)
        if np.array_equal(and_result, a_out):
            def make_bool_and(cc1, cc2, ooc, b):
                def fn(g):
                    a = to_np(g)
                    m1 = a == cc1
                    m2 = a == cc2
                    return to_grid(np.where(m1 & m2, ooc, b))
                return fn
            if all(grid_eq(make_bool_and(c1, c2, oc, bg)(i), o) for i, o in train_pairs):
                rules.append(RuleCandidate(f"bool_and({c1},{c2})→{oc}", make_bool_and(c1, c2, oc, bg), priority=7.0))

        # XOR: symmetric difference
        xor_result = np.where(mask1 ^ mask2, oc, bg)
        if np.array_equal(xor_result, a_out):
            def make_bool_xor(cc1, cc2, ooc, b):
                def fn(g):
                    a = to_np(g)
                    m1 = a == cc1
                    m2 = a == cc2
                    return to_grid(np.where(m1 ^ m2, ooc, b))
                return fn
            if all(grid_eq(make_bool_xor(c1, c2, oc, bg)(i), o) for i, o in train_pairs):
                rules.append(RuleCandidate(f"bool_xor({c1},{c2})→{oc}", make_bool_xor(c1, c2, oc, bg), priority=7.0))

    return rules


def infer_ray_cast_fill(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Fill background cells at row/column intersections of same-colored lines."""
    rules = []
    for bg in range(10):
        def make_ray_cast(bg_val):
            def fn(g):
                a = to_np(g).copy()
                h, w = a.shape
                for r in range(h):
                    for c in range(w):
                        if a[r, c] != bg_val:
                            continue
                        colors = []
                        for dc in range(-1, -c - 1, -1):
                            if a[r, c + dc] != bg_val:
                                colors.append(int(a[r, c + dc]))
                                break
                        for dc in range(1, w - c):
                            if a[r, c + dc] != bg_val:
                                colors.append(int(a[r, c + dc]))
                                break
                        for dr in range(-1, -r - 1, -1):
                            if a[r + dr, c] != bg_val:
                                colors.append(int(a[r + dr, c]))
                                break
                        for dr in range(1, h - r):
                            if a[r + dr, c] != bg_val:
                                colors.append(int(a[r + dr, c]))
                                break
                        if len(colors) >= 2 and len(set(colors)) == 1:
                            a[r, c] = colors[0]
                return to_grid(a)
            return fn
        try:
            fn = make_ray_cast(bg)
            if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(f"ray_cast_fill(bg={bg})", fn, priority=8.0))
        except Exception:
            pass
    return rules


def infer_flood_enclosed(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Fill enclosed background regions with boundary color."""
    rules = []
    for bg in range(10):
        def make_flood(bg_val):
            def fn(g):
                a = to_np(g).copy()
                h, w = a.shape
                visited = np.zeros_like(a, dtype=bool)
                for r in range(h):
                    for c in range(w):
                        if a[r, c] != bg_val or visited[r, c]:
                            continue
                        region = []
                        boundary = []
                        stack = [(r, c)]
                        touches_border = False
                        while stack:
                            rr, cc = stack.pop()
                            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                                touches_border = True
                                continue
                            if visited[rr, cc]:
                                continue
                            if a[rr, cc] != bg_val:
                                boundary.append(int(a[rr, cc]))
                                continue
                            visited[rr, cc] = True
                            region.append((rr, cc))
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                stack.append((rr + dr, cc + dc))
                        if not touches_border and boundary and region:
                            from collections import Counter
                            fill_c = Counter(boundary).most_common(1)[0][0]
                            for rr, cc in region:
                                a[rr, cc] = fill_c
                return to_grid(a)
            return fn
        try:
            fn = make_flood(bg)
            if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(f"flood_enclosed(bg={bg})", fn, priority=7.5))
        except Exception:
            pass
    return rules


def infer_line_between_same_color(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Draw straight lines between pairs of same-colored cells."""
    rules = []
    for bg in range(10):
        def make_lines(bg_val):
            def fn(g):
                a = to_np(g).copy()
                h, w = a.shape
                by_color = {}
                for r in range(h):
                    for c in range(w):
                        if a[r, c] != bg_val:
                            color = int(a[r, c])
                            if color not in by_color:
                                by_color[color] = []
                            by_color[color].append((r, c))
                for color, pts in by_color.items():
                    for i in range(len(pts)):
                        for j in range(i + 1, len(pts)):
                            r1, c1 = pts[i]
                            r2, c2 = pts[j]
                            dr, dc = r2 - r1, c2 - c1
                            if dr == 0:
                                step = 1 if dc > 0 else -1
                                for cc in range(c1 + step, c2, step):
                                    if a[r1, cc] == bg_val:
                                        a[r1, cc] = color
                            elif dc == 0:
                                step = 1 if dr > 0 else -1
                                for rr in range(r1 + step, r2, step):
                                    if a[rr, c1] == bg_val:
                                        a[rr, c1] = color
                            elif abs(dr) == abs(dc):
                                sr = 1 if dr > 0 else -1
                                sc = 1 if dc > 0 else -1
                                for s in range(1, abs(dr)):
                                    rr = r1 + s * sr
                                    cc = c1 + s * sc
                                    if a[rr, cc] == bg_val:
                                        a[rr, cc] = color
                return to_grid(a)
            return fn
        try:
            fn = make_lines(bg)
            if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                rules.append(RuleCandidate(f"lines_between(bg={bg})", fn, priority=7.5))
        except Exception:
            pass
    return rules


def infer_diagonal_continuation(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Detect regular point pattern and extend with a new color."""
    rules = []
    for bg in range(10):
        for ext_c in range(10):
            if ext_c == bg:
                continue
            def make_diag_cont(bg_val, ext_color):
                def fn(g):
                    a = to_np(g).copy()
                    h, w = a.shape
                    by_color = {}
                    for r in range(h):
                        for c in range(w):
                            if a[r, c] != bg_val:
                                color = int(a[r, c])
                                if color not in by_color:
                                    by_color[color] = []
                                by_color[color].append((r, c))
                    for color, pts in by_color.items():
                        if len(pts) < 2:
                            continue
                        pts.sort()
                        dr = pts[1][0] - pts[0][0]
                        dc = pts[1][1] - pts[0][1]
                        if dr == 0 and dc == 0:
                            continue
                        regular = all(
                            pts[i][0] - pts[i - 1][0] == dr and pts[i][1] - pts[i - 1][1] == dc
                            for i in range(2, len(pts))
                        )
                        if regular:
                            rr, cc = pts[-1][0] + dr, pts[-1][1] + dc
                            while 0 <= rr < h and 0 <= cc < w:
                                if a[rr, cc] == bg_val:
                                    a[rr, cc] = ext_color
                                rr += dr
                                cc += dc
                            rr, cc = pts[0][0] - dr, pts[0][1] - dc
                            while 0 <= rr < h and 0 <= cc < w:
                                if a[rr, cc] == bg_val:
                                    a[rr, cc] = ext_color
                                rr -= dr
                                cc -= dc
                    return to_grid(a)
                return fn
            try:
                fn = make_diag_cont(bg, ext_c)
                if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(
                        f"diag_cont(bg={bg},ext={ext_c})", fn, priority=8.0))
            except Exception:
                pass
    return rules


def infer_sudoku_completion(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Fill 0s in NxN grid so each row and column has each color exactly once."""
    rules = []
    inp0 = to_np(train_pairs[0][0])
    out0 = to_np(train_pairs[0][1])
    if inp0.shape != out0.shape: return rules
    h, w = inp0.shape
    if h != w or h > 9 or 0 not in inp0: return rules
    target_colors = set(int(x) for x in out0.flatten())
    if len(target_colors) != h: return rules

    def solve(g, tc):
        a = to_np(g).copy(); n = a.shape[0]
        zeros = [(r, c) for r in range(n) for c in range(n) if a[r, c] == 0]
        def possible(r, c):
            return tc - set(int(x) for x in a[r, :]) - set(int(x) for x in a[:, c]) - {0}
        def bt(idx):
            if idx == len(zeros): return True
            r, c = zeros[idx]
            for color in sorted(possible(r, c)):
                a[r, c] = color
                if bt(idx + 1): return True
                a[r, c] = 0
            return False
        return to_grid(a) if bt(0) else None

    def fn(g):
        result = solve(g, target_colors)
        if result is None: raise ValueError
        return result

    try:
        if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
            rules.append(RuleCandidate("sudoku_completion", fn, priority=9.0))
    except Exception:
        pass
    return rules


def infer_rotate_around_pivot(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Rotate colored line 90° CW/CCW around a pivot cell, recolor original."""
    rules = []
    if not all(to_np(inp).shape == to_np(out).shape for inp, out in train_pairs):
        return rules

    for pivot_color in range(10):
        for recolor in range(10):
            if recolor == pivot_color: continue
            for direction in ['cw', 'ccw']:
                def make_rotate(pc, rc, d):
                    def fn(g):
                        a = to_np(g).copy(); h, w = a.shape
                        pivots = list(zip(*np.where(a == pc)))
                        if len(pivots) != 1: raise ValueError
                        pr, pcc = pivots[0]
                        others = [(r, c, int(a[r, c])) for r in range(h) for c in range(w)
                                  if a[r, c] != 0 and a[r, c] != pc]
                        if not others: raise ValueError
                        orig_color = others[0][2]
                        if not all(c == orig_color for _, _, c in others): raise ValueError
                        result = np.zeros_like(a)
                        result[pr, pcc] = pc
                        for r, c, _ in others:
                            result[r, c] = rc
                        for r, c, _ in others:
                            dr, dc = r - pr, c - pcc
                            if d == 'cw':
                                nr, nc = pr + dc, pcc - dr
                            else:
                                nr, nc = pr - dc, pcc + dr
                            if 0 <= nr < h and 0 <= nc < w:
                                result[nr, nc] = orig_color
                        return to_grid(result)
                    return fn
                try:
                    fn = make_rotate(pivot_color, recolor, direction)
                    if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                        rules.append(RuleCandidate(
                            f"rotate_{direction}(pivot={pivot_color},recolor={recolor})",
                            fn, priority=8.5))
                except Exception:
                    pass
    return rules


def infer_path_routing(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Trace L-path with CW/CCW turn markers."""
    rules = []
    if not all(to_np(inp).shape == to_np(out).shape for inp, out in train_pairs):
        return rules

    _CW = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}
    _CCW = {(0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0), (1, 0): (0, 1)}

    for path_color in range(10):
        for cw_color in range(10):
            for ccw_color in range(10):
                if len({path_color, cw_color, ccw_color}) < 3: continue

                def make_path(pc, cwc, ccwc):
                    def fn(g):
                        a = to_np(g).copy(); h, w = a.shape
                        starts = list(zip(*np.where(a == pc)))
                        if len(starts) != 1: raise ValueError
                        sr, sc = starts[0]
                        targets = {}
                        for r in range(h):
                            for c in range(w):
                                if a[r, c] != 0 and a[r, c] != pc:
                                    targets[(r, c)] = int(a[r, c])
                        if not targets: raise ValueError
                        dr, dc = 0, 1
                        cr, cc = sr, sc
                        for _ in range(200):
                            nr, nc = cr + dr, cc + dc
                            if (nr, nc) in targets:
                                tc = targets[(nr, nc)]
                                del targets[(nr, nc)]
                                if tc == cwc:
                                    dr, dc = _CW[(dr, dc)]
                                elif tc == ccwc:
                                    dr, dc = _CCW[(dr, dc)]
                                else:
                                    raise ValueError
                                continue
                            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                                break
                            a[nr, nc] = pc
                            cr, cc = nr, nc
                        return to_grid(a)
                    return fn

                try:
                    fn = make_path(path_color, cw_color, ccw_color)
                    if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
                        rules.append(RuleCandidate(
                            f"path_route({path_color},cw={cw_color},ccw={ccw_color})",
                            fn, priority=8.5))
                        return rules  # expensive search, return on first match
                except Exception:
                    pass
    return rules


def infer_diag_perp_extension(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Connect longest diagonal pair, extend on-line markers perpendicular."""
    rules = []
    if not all(to_np(inp).shape == to_np(out).shape for inp, out in train_pairs):
        return rules

    def fn(g):
        a = to_np(g).copy(); h, w = a.shape
        bg = int(np.bincount(a.flatten()).argmax())
        from itertools import combinations as _comb
        all_pts = [(r, c, int(a[r, c])) for r in range(h) for c in range(w) if a[r, c] != bg]
        best_pair = None; best_len = 0
        for (r1, c1, col1), (r2, c2, col2) in _comb(all_pts, 2):
            if col1 != col2: continue
            dr, dc = r2 - r1, c2 - c1
            if dr == 0 or dc == 0 or abs(dr) != abs(dc): continue
            if abs(dr) > best_len:
                best_len = abs(dr)
                best_pair = (r1, c1, r2, c2, col1)
        if best_pair is None: raise ValueError
        r1, c1, r2, c2, line_color = best_pair
        sr = 1 if r2 - r1 > 0 else -1
        sc = 1 if c2 - c1 > 0 else -1
        steps = abs(r2 - r1)
        for s in range(1, steps):
            rr, cc = r1 + s * sr, c1 + s * sc
            if 0 <= rr < h and 0 <= cc < w and a[rr, cc] == bg:
                a[rr, cc] = line_color
        for s in range(0, steps + 1):
            rr, cc = r1 + s * sr, c1 + s * sc
            if 0 <= rr < h and 0 <= cc < w:
                cell_color = int(a[rr, cc])
                if cell_color != bg and cell_color != line_color:
                    for pdr, pdc in [(-sc, sr), (sc, -sr)]:
                        pr, pc = rr + pdr, cc + pdc
                        while 0 <= pr < h and 0 <= pc < w:
                            if a[pr, pc] == bg:
                                a[pr, pc] = cell_color
                            elif a[pr, pc] != cell_color:
                                break
                            pr += pdr; pc += pdc
        return to_grid(a)

    try:
        if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
            rules.append(RuleCandidate("diag_perp_extend", fn, priority=8.0))
    except Exception:
        pass
    return rules


def infer_separator_center(train_pairs: List[Tuple[Grid, Grid]]) -> List[RuleCandidate]:
    """Place most-common color above separator at bottom center."""
    rules = []

    def find_sep_row(a):
        h, w = a.shape
        for r in range(h):
            vals = set(a[r, :])
            if len(vals) == 1 and list(vals)[0] != 0:
                return r, int(list(vals)[0])
        return None, None

    def fn(g):
        a = to_np(g).copy(); h, w = a.shape
        sep_row, sep_color = find_sep_row(a)
        if sep_row is None: raise ValueError
        above_colors = []
        for r in range(sep_row):
            for c in range(w):
                if a[r, c] != 0 and a[r, c] != sep_color:
                    above_colors.append(int(a[r, c]))
        if not above_colors: raise ValueError
        from collections import Counter as _Counter
        most_common = _Counter(above_colors).most_common(1)[0][0]
        a[h - 1, w // 2] = most_common
        return to_grid(a)

    try:
        if all(grid_eq(fn(inp), out) for inp, out in train_pairs):
            rules.append(RuleCandidate("separator_center", fn, priority=8.0))
    except Exception:
        pass
    return rules


def infer_composition(train_pairs: List[Tuple[Grid, Grid]], base_rules: List[RuleCandidate]) -> List[RuleCandidate]:
    """Try composing two rules: rule2(rule1(input))."""
    rules = []
    # Only compose rules that partially work (transform input to something reasonable)
    for r1 in base_rules[:15]:  # Limit search
        for r2 in base_rules[:15]:
            if r1.name == r2.name:
                continue
            def make_compose(fn1, fn2):
                def fn(g):
                    intermediate = fn1(g)
                    if intermediate is None:
                        return None
                    return fn2(intermediate)
                return fn

            composed_fn = make_compose(r1.apply, r2.apply)
            try:
                if all(grid_eq(composed_fn(inp), out) for inp, out in train_pairs):
                    rules.append(RuleCandidate(
                        f"{r2.name}({r1.name}(x))",
                        composed_fn, priority=min(r1.priority, r2.priority) - 1.0))
            except Exception:
                pass

    return rules


# ─── Main synthesis engine ───────────────────────────────────────────────────

class ARCSynthesizer:
    """Program synthesis engine for ARC tasks."""

    def __init__(self, max_time: float = 30.0, verbose: bool = False):
        self.max_time = max_time
        self.verbose = verbose

    def synthesize(self, task: dict) -> List[Grid]:
        """Find transformation rules and apply to test input."""
        start = time.time()

        train_pairs = [(ex['input'], ex['output']) for ex in task['train']]
        test_inputs = [ex['input'] for ex in task['test']]

        if self.verbose:
            print(f"  Training pairs: {len(train_pairs)}")
            for i, (inp, out) in enumerate(train_pairs):
                print(f"    Pair {i}: {grid_shape(inp)} → {grid_shape(out)}")

        # Phase 1: Try direct rule inference
        candidates = []

        # Identity check
        if all(grid_eq(inp, out) for inp, out in train_pairs):
            candidates.append(RuleCandidate("identity", lambda g: g, priority=10.0))

        # Color mapping
        cm = infer_color_mapping(train_pairs)
        if cm:
            candidates.append(cm)

        # Geometric transforms
        candidates.extend(infer_geometric_transform(train_pairs))

        # Crop operations
        candidates.extend(infer_crop(train_pairs))

        # Scaling
        candidates.extend(infer_scaling(train_pairs))

        # Tiling
        candidates.extend(infer_tiling(train_pairs))

        # Gravity
        candidates.extend(infer_gravity(train_pairs))

        # Fill enclosed
        candidates.extend(infer_fill(train_pairs))

        # Border operations
        candidates.extend(infer_border_ops(train_pairs))

        # Majority vote
        candidates.extend(infer_majority_vote(train_pairs))

        # XOR overlay
        candidates.extend(infer_xor_overlay(train_pairs))

        # Subgrid extraction
        candidates.extend(infer_subgrid_extraction(train_pairs))

        # Unique subgrid
        candidates.extend(infer_unique_subgrid(train_pairs))

        # Object recoloring
        candidates.extend(infer_object_recolor(train_pairs))

        # Color-based scaling
        candidates.extend(infer_color_scale(train_pairs))

        # Symmetry completion
        candidates.extend(infer_symmetry_completion(train_pairs))

        # Line extension
        candidates.extend(infer_line_extension(train_pairs))

        # Object stamping
        candidates.extend(infer_object_stamp(train_pairs))

        # Mirror halves
        candidates.extend(infer_mirror_halves(train_pairs))

        # Color removal / keeping
        candidates.extend(infer_remove_color(train_pairs))
        candidates.extend(infer_keep_color(train_pairs))

        # Row/column operations
        candidates.extend(infer_row_col_ops(train_pairs))

        # Split + recombine (OR/AND overlay)
        candidates.extend(infer_split_recombine(train_pairs))

        # Flood fill from markers
        candidates.extend(infer_flood_from_markers(train_pairs))

        # Pattern replacement
        candidates.extend(infer_replace_pattern(train_pairs))

        # Wallpaper tiling
        candidates.extend(infer_wallpaper_tiling(train_pairs))

        # Grid segmentation (split by divider lines, recombine)
        candidates.extend(infer_grid_segmentation(train_pairs))

        # Diagonal flips
        candidates.extend(infer_diagonal_flip(train_pairs))

        # Boolean operations between color layers
        candidates.extend(infer_boolean_ops(train_pairs))

        # Unique subgrid among repeated blocks
        candidates.extend(infer_unique_subgrid(train_pairs))

        # Ray cast fill (intersecting row/column lines)
        candidates.extend(infer_ray_cast_fill(train_pairs))

        # Flood fill enclosed regions
        candidates.extend(infer_flood_enclosed(train_pairs))

        # Lines between same-colored cells
        candidates.extend(infer_line_between_same_color(train_pairs))

        # Diagonal/linear pattern continuation
        if time.time() - start < self.max_time * 0.3:
            candidates.extend(infer_diagonal_continuation(train_pairs))

        # Sudoku-like completion
        candidates.extend(infer_sudoku_completion(train_pairs))

        # Rotate around pivot
        if time.time() - start < self.max_time * 0.3:
            candidates.extend(infer_rotate_around_pivot(train_pairs))

        # Diagonal + perpendicular extension
        candidates.extend(infer_diag_perp_extension(train_pairs))

        # Separator + center placement
        candidates.extend(infer_separator_center(train_pairs))

        # Path routing with CW/CCW markers
        if time.time() - start < self.max_time * 0.4 and not candidates:
            candidates.extend(infer_path_routing(train_pairs))

        # Per-cell rules
        if time.time() - start < self.max_time * 0.5:
            candidates.extend(infer_per_cell_rule(train_pairs))

        # Pattern repeat
        candidates.extend(infer_pattern_repeat(train_pairs))

        if self.verbose:
            print(f"  Phase 1: {len(candidates)} candidates found")

        # Phase 2: Try compositions if no direct solution
        if not candidates and time.time() - start < self.max_time * 0.7:
            # Build base transforms to compose
            base_transforms = []
            for name, fn in [
                ("rotate90", rotate90), ("rotate180", rotate180), ("rotate270", rotate270),
                ("flip_h", flip_h), ("flip_v", flip_v), ("transpose", transpose),
            ]:
                base_transforms.append(RuleCandidate(name, fn))

            for bg in [0, background_color(train_pairs[0][0])]:
                base_transforms.append(RuleCandidate(
                    f"crop(bg={bg})", lambda g, b=bg: crop_to_content(g, bg=b)))

            for factor in [2, 3]:
                base_transforms.append(RuleCandidate(
                    f"scale({factor})", lambda g, f=factor: scale_up(g, f)))

            candidates.extend(infer_composition(train_pairs, base_transforms))

            if self.verbose:
                print(f"  Phase 2 (composition): {len(candidates)} candidates")

        # Sort by priority
        candidates.sort(key=lambda r: r.priority, reverse=True)

        # Apply best candidate to test inputs
        predictions = []
        for test_inp in test_inputs:
            pred = None
            for rule in candidates:
                result = rule.apply(test_inp)
                if result is not None:
                    pred = result
                    if self.verbose:
                        print(f"  Applied: {rule.name}")
                    break
            predictions.append(pred if pred else test_inp)

        return predictions


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(data_dir: str, max_tasks: int = 0, verbose: bool = False):
    """Evaluate synthesizer on ARC tasks."""
    task_files = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    if max_tasks > 0:
        task_files = task_files[:max_tasks]

    synth = ARCSynthesizer(max_time=30.0, verbose=verbose)
    solved = 0
    total = 0

    results = {}

    for i, tf in enumerate(task_files):
        task_id = os.path.basename(tf).replace('.json', '')
        with open(tf) as f:
            task = json.load(f)

        try:
            predictions = synth.synthesize(task)
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            predictions = [task['test'][j]['input'] for j in range(len(task['test']))]

        task_solved = True
        for j, test_ex in enumerate(task['test']):
            total += 1
            if 'output' in test_ex and grid_eq(predictions[j], test_ex['output']):
                solved += 1
            else:
                task_solved = False

        results[task_id] = task_solved

        if (i + 1) % 30 == 0 or i == len(task_files) - 1:
            print(f"  [{i+1}/{len(task_files)}] solved={sum(results.values())}/{i+1} "
                  f"({100*sum(results.values())/(i+1):.1f}%)")

    print(f"\n{'='*50}")
    print(f"Program Synthesis Results: {sum(results.values())}/{len(task_files)} "
          f"({100*sum(results.values())/len(task_files):.1f}%)")
    print(f"{'='*50}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ARC Program Synthesis')
    parser.add_argument('--data', default='ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation',
                        help='Path to ARC task directory')
    parser.add_argument('--max-tasks', type=int, default=0, help='Max tasks to evaluate')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', type=str, default='', help='Save results to JSON')
    args = parser.parse_args()

    print(f"Evaluating on: {args.data}")
    results = evaluate(args.data, max_tasks=args.max_tasks, verbose=args.verbose)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
