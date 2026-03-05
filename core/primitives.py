"""
Compositional Reasoning over Novel Primitives

A library of atomic grid-transformation primitives that can be
composed into multi-step programs. Each primitive is a pure function
grid → grid (or grid → value) with typed parameters.

The composition engine searches over programs of depth 1-3 by
chaining primitives, testing each candidate against training examples.

Key insight: ARC tasks are compositional — most solutions are 1-3
primitives chained together. By having a rich primitive bank and
efficient search, we can solve tasks that no single operation handles.
"""

from typing import List, Dict, Tuple, Optional, Callable, Any, Set
from collections import Counter
import copy


# ============================================================================
# Type Aliases
# ============================================================================
Grid = List[List[int]]
Primitive = Callable[..., Grid]


# ============================================================================
# Primitive Bank — Atomic Operations
# ============================================================================

def p_identity(grid: Grid) -> Grid:
    return [row[:] for row in grid]

def p_rotate_90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]

def p_rotate_180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]

def p_rotate_270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*[r[::-1] for r in grid])]

def p_flip_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]

def p_flip_v(grid: Grid) -> Grid:
    return grid[::-1]

def p_transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]

def p_gravity_down(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    bg = _bg(grid)
    result = [[bg]*cols for _ in range(rows)]
    for c in range(cols):
        vals = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
        for i, v in enumerate(vals):
            result[rows - len(vals) + i][c] = v
    return result

def p_gravity_up(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    bg = _bg(grid)
    result = [[bg]*cols for _ in range(rows)]
    for c in range(cols):
        vals = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
        for i, v in enumerate(vals):
            result[i][c] = v
    return result

def p_gravity_left(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    bg = _bg(grid)
    result = [[bg]*cols for _ in range(rows)]
    for r in range(rows):
        vals = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
        for i, v in enumerate(vals):
            result[r][i] = v
    return result

def p_gravity_right(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    bg = _bg(grid)
    result = [[bg]*cols for _ in range(rows)]
    for r in range(rows):
        vals = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
        for i, v in enumerate(vals):
            result[r][cols - len(vals) + i] = v
    return result

def p_remove_bg(grid: Grid) -> Grid:
    """Crop to bounding box of non-background cells."""
    bg = _bg(grid)
    rows, cols = len(grid), len(grid[0])
    r1, r2, c1, c2 = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                r1 = min(r1, r)
                r2 = max(r2, r)
                c1 = min(c1, c)
                c2 = max(c2, c)
    if r2 < r1:
        return grid
    return [row[c1:c2+1] for row in grid[r1:r2+1]]

def p_fill_holes(grid: Grid) -> Grid:
    """Fill background cells that are enclosed by non-background cells."""
    rows, cols = len(grid), len(grid[0])
    bg = _bg(grid)
    # Flood fill from edges to find non-enclosed bg
    visited = [[False]*cols for _ in range(rows)]
    queue = []
    for r in range(rows):
        for c in [0, cols-1]:
            if grid[r][c] == bg and not visited[r][c]:
                queue.append((r, c))
                visited[r][c] = True
    for c in range(cols):
        for r in [0, rows-1]:
            if grid[r][c] == bg and not visited[r][c]:
                queue.append((r, c))
                visited[r][c] = True
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == bg:
                visited[nr][nc] = True
                queue.append((nr, nc))
    result = [row[:] for row in grid]
    # Find the most common non-bg color to fill with
    non_bg = [c for row in grid for c in row if c != bg]
    fill_color = Counter(non_bg).most_common(1)[0][0] if non_bg else 1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg and not visited[r][c]:
                result[r][c] = fill_color
    return result

def p_extract_largest(grid: Grid) -> Grid:
    """Extract bounding box of the largest connected object."""
    bg = _bg(grid)
    objects = _find_objects(grid, bg)
    if not objects:
        return grid
    largest = max(objects, key=lambda o: o['size'])
    r1, c1, r2, c2 = largest['bbox']
    return [row[c1:c2+1] for row in grid[r1:r2+1]]

def p_extract_smallest(grid: Grid) -> Grid:
    """Extract bounding box of the smallest connected object."""
    bg = _bg(grid)
    objects = _find_objects(grid, bg)
    if not objects:
        return grid
    smallest = min(objects, key=lambda o: o['size'])
    r1, c1, r2, c2 = smallest['bbox']
    return [row[c1:c2+1] for row in grid[r1:r2+1]]

def p_sort_rows(grid: Grid) -> Grid:
    """Sort each row by color value."""
    return [sorted(row) for row in grid]

def p_sort_cols(grid: Grid) -> Grid:
    """Sort each column by color value."""
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for c in range(cols):
        col_vals = sorted(grid[r][c] for r in range(rows))
        for r in range(rows):
            result[r][c] = col_vals[r]
    return result

def p_unique_rows(grid: Grid) -> Grid:
    """Remove duplicate rows."""
    seen = []
    result = []
    for row in grid:
        key = tuple(row)
        if key not in seen:
            seen.append(key)
            result.append(row[:])
    return result if result else grid

def p_unique_cols(grid: Grid) -> Grid:
    """Remove duplicate columns."""
    cols = len(grid[0]) if grid else 0
    seen = []
    keep = []
    for c in range(cols):
        col = tuple(grid[r][c] for r in range(len(grid)))
        if col not in seen:
            seen.append(col)
            keep.append(c)
    return [[grid[r][c] for c in keep] for r in range(len(grid))] if keep else grid

def p_mirror_h_complete(grid: Grid) -> Grid:
    """Mirror non-bg cells horizontally to complete symmetry."""
    bg = _bg(grid)
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            mc = cols - 1 - c
            if grid[r][c] != bg and result[r][mc] == bg:
                result[r][mc] = grid[r][c]
            elif grid[r][mc] != bg and result[r][c] == bg:
                result[r][c] = grid[r][mc]
    return result

def p_mirror_v_complete(grid: Grid) -> Grid:
    """Mirror non-bg cells vertically to complete symmetry."""
    bg = _bg(grid)
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        mr = rows - 1 - r
        for c in range(cols):
            if grid[r][c] != bg and result[mr][c] == bg:
                result[mr][c] = grid[r][c]
            elif grid[mr][c] != bg and result[r][c] == bg:
                result[r][c] = grid[mr][c]
    return result

def p_invert(grid: Grid) -> Grid:
    """Swap background and most common foreground color."""
    bg = _bg(grid)
    non_bg = [c for row in grid for c in row if c != bg]
    if not non_bg:
        return grid
    fg = Counter(non_bg).most_common(1)[0][0]
    return [[fg if c == bg else (bg if c == fg else c) for c in row] for row in grid]

def p_outline(grid: Grid) -> Grid:
    """Keep only border cells of each object (hollow out)."""
    bg = _bg(grid)
    rows, cols = len(grid), len(grid[0])
    result = [[bg]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                is_border = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols or grid[nr][nc] != grid[r][c]:
                        is_border = True
                        break
                if is_border:
                    result[r][c] = grid[r][c]
    return result

def p_dilate(grid: Grid) -> Grid:
    """Expand each non-bg cell to its 4-neighbors."""
    bg = _bg(grid)
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == bg:
                        result[nr][nc] = grid[r][c]
    return result


def p_flood_fill_enclosed(grid: Grid) -> Grid:
    """Fill enclosed background regions with the enclosing object's color."""
    bg = _bg(grid)
    rows, cols = len(grid), len(grid[0])
    # Find bg cells reachable from edges
    edge_bg = set()
    queue = []
    for r in range(rows):
        for c in [0, cols - 1]:
            if grid[r][c] == bg:
                edge_bg.add((r, c))
                queue.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if grid[r][c] == bg and (r, c) not in edge_bg:
                edge_bg.add((r, c))
                queue.append((r, c))
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in edge_bg and grid[nr][nc] == bg:
                edge_bg.add((nr, nc))
                queue.append((nr, nc))
    # Fill non-edge bg with nearest non-bg color
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg and (r, c) not in edge_bg:
                # Find nearest non-bg neighbor color
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != bg:
                        result[r][c] = grid[nr][nc]
                        break
    return result


def p_replace_bg_with_most_common(grid: Grid) -> Grid:
    """Replace background cells with the most common foreground color."""
    bg = _bg(grid)
    non_bg = [c for row in grid for c in row if c != bg]
    if not non_bg:
        return grid
    fg = Counter(non_bg).most_common(1)[0][0]
    return [[fg if c == bg else c for c in row] for row in grid]


def p_keep_largest_color(grid: Grid) -> Grid:
    """Keep only the most common non-bg color, rest becomes bg."""
    bg = _bg(grid)
    non_bg = [c for row in grid for c in row if c != bg]
    if not non_bg:
        return grid
    keep = Counter(non_bg).most_common(1)[0][0]
    return [[c if c == keep else bg for c in row] for row in grid]


def p_keep_minority_color(grid: Grid) -> Grid:
    """Keep only the least common non-bg color, rest becomes bg."""
    bg = _bg(grid)
    non_bg = [c for row in grid for c in row if c != bg]
    if not non_bg:
        return grid
    counts = Counter(non_bg)
    keep = counts.most_common()[-1][0]
    return [[c if c == keep else bg for c in row] for row in grid]


def p_top_half(grid: Grid) -> Grid:
    """Return top half of grid."""
    h = len(grid)
    return [row[:] for row in grid[:h // 2]]


def p_bottom_half(grid: Grid) -> Grid:
    """Return bottom half of grid."""
    h = len(grid)
    return [row[:] for row in grid[h // 2:]]


def p_left_half(grid: Grid) -> Grid:
    """Return left half of grid."""
    w = len(grid[0]) if grid else 0
    return [row[:w // 2] for row in grid]


def p_right_half(grid: Grid) -> Grid:
    """Return right half of grid."""
    w = len(grid[0]) if grid else 0
    return [row[w // 2:] for row in grid]


def p_xor_halves_h(grid: Grid) -> Grid:
    """XOR top and bottom halves: keep cells that differ."""
    h = len(grid)
    w = len(grid[0]) if grid else 0
    half = h // 2
    if h % 2 != 0 or half == 0:
        return grid
    bg = _bg(grid)
    result = []
    for r in range(half):
        row = []
        for c in range(w):
            top = grid[r][c]
            bot = grid[r + half][c]
            if top != bot:
                row.append(top if top != bg else bot)
            else:
                row.append(bg)
        result.append(row)
    return result


def p_xor_halves_v(grid: Grid) -> Grid:
    """XOR left and right halves: keep cells that differ."""
    h = len(grid)
    w = len(grid[0]) if grid else 0
    half = w // 2
    if w % 2 != 0 or half == 0:
        return grid
    bg = _bg(grid)
    result = []
    for r in range(h):
        row = []
        for c in range(half):
            left = grid[r][c]
            right = grid[r][c + half]
            if left != right:
                row.append(left if left != bg else right)
            else:
                row.append(bg)
        result.append(row)
    return result


def p_and_halves_h(grid: Grid) -> Grid:
    """AND top and bottom halves: keep cells that match (non-bg)."""
    h = len(grid)
    w = len(grid[0]) if grid else 0
    half = h // 2
    if h % 2 != 0 or half == 0:
        return grid
    bg = _bg(grid)
    result = []
    for r in range(half):
        row = []
        for c in range(w):
            top = grid[r][c]
            bot = grid[r + half][c]
            if top != bg and bot != bg:
                row.append(top)
            else:
                row.append(bg)
        result.append(row)
    return result


def p_and_halves_v(grid: Grid) -> Grid:
    """AND left and right halves: keep cells present in both."""
    h = len(grid)
    w = len(grid[0]) if grid else 0
    half = w // 2
    if w % 2 != 0 or half == 0:
        return grid
    bg = _bg(grid)
    result = []
    for r in range(h):
        row = []
        for c in range(half):
            left = grid[r][c]
            right = grid[r][c + half]
            if left != bg and right != bg:
                row.append(left)
            else:
                row.append(bg)
        result.append(row)
    return result


def p_or_halves_h(grid: Grid) -> Grid:
    """OR top and bottom halves: overlay non-bg cells from both."""
    h = len(grid)
    w = len(grid[0]) if grid else 0
    half = h // 2
    if h % 2 != 0 or half == 0:
        return grid
    bg = _bg(grid)
    result = []
    for r in range(half):
        row = []
        for c in range(w):
            top = grid[r][c]
            bot = grid[r + half][c]
            if top != bg:
                row.append(top)
            elif bot != bg:
                row.append(bot)
            else:
                row.append(bg)
        result.append(row)
    return result


def p_or_halves_v(grid: Grid) -> Grid:
    """OR left and right halves: overlay non-bg cells from both."""
    h = len(grid)
    w = len(grid[0]) if grid else 0
    half = w // 2
    if w % 2 != 0 or half == 0:
        return grid
    bg = _bg(grid)
    result = []
    for r in range(h):
        row = []
        for c in range(half):
            left = grid[r][c]
            right = grid[r][c + half]
            if left != bg:
                row.append(left)
            elif right != bg:
                row.append(right)
            else:
                row.append(bg)
        result.append(row)
    return result


def p_count_colors_per_row(grid: Grid) -> Grid:
    """Replace each cell with the count of distinct non-bg colors in its row."""
    bg = _bg(grid)
    result = []
    for row in grid:
        n = len({c for c in row if c != bg})
        result.append([n] * len(row))
    return result


def p_replace_each_object_with_color(grid: Grid) -> Grid:
    """Replace each connected object with a single solid color (its own)."""
    bg = _bg(grid)
    objs = _find_objects(grid, bg)
    result = [[bg] * len(grid[0]) for _ in range(len(grid))]
    for obj in objs:
        r1, c1, r2, c2 = obj['bbox']
        color = obj['color']
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                result[r][c] = color
    return result


def p_erode(grid: Grid) -> Grid:
    """Remove non-bg cells that touch background (opposite of dilate)."""
    bg = _bg(grid)
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols or grid[nr][nc] == bg:
                        result[r][c] = bg
                        break
    return result


def p_upscale_2x(grid: Grid) -> Grid:
    """Upscale grid by 2x (each cell becomes 2x2 block)."""
    result = []
    for row in grid:
        new_row = []
        for c in row:
            new_row.extend([c, c])
        result.append(new_row[:])
        result.append(new_row[:])
    return result


def p_downscale_2x(grid: Grid) -> Grid:
    """Downscale grid by 2x (take majority of each 2x2 block)."""
    rows, cols = len(grid), len(grid[0])
    if rows < 2 or cols < 2:
        return grid
    result = []
    for r in range(0, rows - 1, 2):
        new_row = []
        for c in range(0, cols - 1, 2):
            block = [grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]]
            new_row.append(Counter(block).most_common(1)[0][0])
        result.append(new_row)
    return result if result else grid


# ============================================================================
# Parameterized Primitives (generators)
# ============================================================================

def make_scale(factor: int) -> Callable:
    def p_scale(grid: Grid) -> Grid:
        result = []
        for row in grid:
            new_row = []
            for c in row:
                new_row.extend([c] * factor)
            for _ in range(factor):
                result.append(new_row[:])
        return result
    p_scale.__name__ = f'scale_{factor}x'
    return p_scale

def make_tile(tr: int, tc: int) -> Callable:
    def p_tile(grid: Grid) -> Grid:
        tiled = []
        for _ in range(tr):
            for row in grid:
                tiled.append(row * tc)
        return tiled
    p_tile.__name__ = f'tile_{tr}x{tc}'
    return p_tile

def make_color_swap(a: int, b: int) -> Callable:
    def p_swap(grid: Grid) -> Grid:
        return [[b if c == a else (a if c == b else c) for c in row] for row in grid]
    p_swap.__name__ = f'swap_{a}_{b}'
    return p_swap


# ============================================================================
# Composition Engine
# ============================================================================

# All zero-argument primitives
BASE_PRIMITIVES: List[Tuple[str, Callable]] = [
    ('identity', p_identity),
    ('rotate_90', p_rotate_90),
    ('rotate_180', p_rotate_180),
    ('rotate_270', p_rotate_270),
    ('flip_h', p_flip_h),
    ('flip_v', p_flip_v),
    ('transpose', p_transpose),
    ('gravity_down', p_gravity_down),
    ('gravity_up', p_gravity_up),
    ('gravity_left', p_gravity_left),
    ('gravity_right', p_gravity_right),
    ('crop', p_remove_bg),
    ('fill_holes', p_fill_holes),
    ('flood_fill_enclosed', p_flood_fill_enclosed),
    ('extract_largest', p_extract_largest),
    ('extract_smallest', p_extract_smallest),
    ('sort_rows', p_sort_rows),
    ('sort_cols', p_sort_cols),
    ('unique_rows', p_unique_rows),
    ('unique_cols', p_unique_cols),
    ('mirror_h', p_mirror_h_complete),
    ('mirror_v', p_mirror_v_complete),
    ('invert', p_invert),
    ('outline', p_outline),
    ('dilate', p_dilate),
    ('erode', p_erode),
    ('replace_bg_fg', p_replace_bg_with_most_common),
    ('keep_largest_color', p_keep_largest_color),
    ('keep_minority_color', p_keep_minority_color),
    ('top_half', p_top_half),
    ('bottom_half', p_bottom_half),
    ('left_half', p_left_half),
    ('right_half', p_right_half),
    ('xor_halves_h', p_xor_halves_h),
    ('xor_halves_v', p_xor_halves_v),
    ('and_halves_h', p_and_halves_h),
    ('and_halves_v', p_and_halves_v),
    ('or_halves_h', p_or_halves_h),
    ('or_halves_v', p_or_halves_v),
    ('solid_objects', p_replace_each_object_with_color),
    ('upscale_2x', p_upscale_2x),
    ('downscale_2x', p_downscale_2x),
]


class CompositionEngine:
    """
    Searches over compositions of primitives to find programs
    that transform all training inputs to their outputs.
    """

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.primitives = list(BASE_PRIMITIVES)
        # Add parameterized variants
        for f in [2, 3, 4]:
            self.primitives.append((f'scale_{f}x', make_scale(f)))
        for tr in range(1, 4):
            for tc in range(1, 4):
                if tr == 1 and tc == 1:
                    continue
                self.primitives.append((f'tile_{tr}x{tc}', make_tile(tr, tc)))

    def search(
        self,
        examples: List[Dict],
        max_depth: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for programs that solve all examples.
        Uses iterative deepening with output-hash pruning:
        intermediate results that duplicate earlier states are skipped.
        
        Returns list of solutions, each containing:
          - program: list of primitive names
          - depth: number of chained primitives
        """
        import time as _time
        depth = max_depth or self.max_depth
        solutions = []
        t0 = _time.time()
        timeout = 60.0  # seconds

        # Target output signature for dimension-based pruning
        target_dims = set()
        for ex in examples:
            oh, ow = len(ex['output']), len(ex['output'][0]) if ex['output'] else 0
            target_dims.add((oh, ow))

        def _grid_hash(g):
            return tuple(tuple(r) for r in g)

        def _dims_match_target(g) -> bool:
            """Check if grid dimensions could lead to target output."""
            h, w = len(g), len(g[0]) if g else 0
            # Allow if dims already match target, or are larger (can be cropped)
            return any(h >= th and w >= tw or (h, w) == (th, tw) for th, tw in target_dims)

        # Precompute depth-1 results per example for pruning
        # Maps: example_idx -> list of (name, fn, result_hash)
        d1_cache: Dict[int, Dict[str, Any]] = {}
        for ei, ex in enumerate(examples):
            d1_cache[ei] = {}
            for name, fn in self.primitives:
                try:
                    r = fn(ex['input'])
                    d1_cache[ei][name] = r
                except Exception:
                    pass

        # Depth 1: single primitives
        for name, fn in self.primitives:
            if self._test_program([fn], examples):
                solutions.append({'program': [name], 'depth': 1})

        if solutions or depth < 2:
            return solutions

        # Depth 2: pairs (with hash dedup)
        seen_d2: Set[Any] = set()
        for n1, f1 in self.primitives:
            if n1 == 'identity':
                continue
            # Quick prune: skip if first step does nothing on example 0
            if 0 in d1_cache and n1 in d1_cache[0]:
                r1 = d1_cache[0][n1]
                if r1 == examples[0]['input']:
                    continue
            for n2, f2 in self.primitives:
                if n2 == 'identity':
                    continue
                if self._test_program([f1, f2], examples):
                    solutions.append({'program': [n1, n2], 'depth': 2})

        if solutions or depth < 3:
            return solutions

        # Depth 3+: iterative deepening with aggressive pruning
        # Build frontier: intermediate results after applying 1 primitive
        # Each entry: (program_names, per-example intermediate grids)
        frontier = []
        for n1, f1 in self.primitives:
            if n1 == 'identity':
                continue
            intermediates = []
            valid = True
            for ei, ex in enumerate(examples):
                if n1 in d1_cache.get(ei, {}):
                    intermediates.append(d1_cache[ei][n1])
                else:
                    valid = False
                    break
            if valid and intermediates[0] != examples[0]['input']:
                frontier.append(([n1], intermediates))

        for current_depth in range(2, depth + 1):
            if _time.time() - t0 > timeout:
                break
            next_frontier = []
            seen_hashes: Set[Any] = set()

            for prog_names, inter_grids in frontier:
                if _time.time() - t0 > timeout:
                    break
                for name, fn in self.primitives:
                    if name == 'identity':
                        continue
                    new_grids = []
                    valid = True
                    for ei, g in enumerate(inter_grids):
                        try:
                            r = fn(g)
                            new_grids.append(r)
                        except Exception:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Check if this solves all examples
                    new_prog = prog_names + [name]
                    if all(new_grids[ei] == examples[ei]['output'] for ei in range(len(examples))):
                        solutions.append({'program': new_prog, 'depth': current_depth})
                        return solutions  # Return first solution at this depth

                    # Prune: skip if we've seen this exact intermediate state
                    state_hash = tuple(_grid_hash(g) for g in new_grids)
                    if state_hash in seen_hashes:
                        continue
                    seen_hashes.add(state_hash)

                    # Prune: skip if dimensions can't reach target
                    if not _dims_match_target(new_grids[0]):
                        continue

                    if current_depth < depth:
                        next_frontier.append((new_prog, new_grids))

            frontier = next_frontier

        return solutions

    def apply_program(self, program_names: List[str], grid: Grid) -> Grid:
        """Apply a named program to a grid."""
        prim_map = {n: f for n, f in self.primitives}
        result = [row[:] for row in grid]
        for name in program_names:
            if name in prim_map:
                result = prim_map[name](result)
        return result

    def _test_program(self, fns: List[Callable], examples: List[Dict]) -> bool:
        """Test if a chain of functions solves all examples."""
        for ex in examples:
            try:
                result = ex['input']
                for fn in fns:
                    result = fn(result)
                if result != ex['output']:
                    return False
            except Exception:
                return False
        return True


# ============================================================================
# Helpers
# ============================================================================

def _bg(grid: Grid) -> int:
    counts = Counter(c for row in grid for c in row)
    return counts.most_common(1)[0][0] if counts else 0

def _find_objects(grid: Grid, bg: int) -> List[Dict]:
    rows, cols = len(grid), len(grid[0]) if grid else 0
    visited = [[False]*cols for _ in range(rows)]
    objects = []
    def flood(r, c, color):
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc] or grid[cr][cc] != color:
                continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                cells = flood(r, c, grid[r][c])
                if cells:
                    min_r = min(cr for cr, _ in cells)
                    max_r = max(cr for cr, _ in cells)
                    min_c = min(cc for _, cc in cells)
                    max_c = max(cc for _, cc in cells)
                    objects.append({
                        'color': grid[r][c], 'cells': cells,
                        'bbox': (min_r, min_c, max_r, max_c),
                        'size': len(cells),
                    })
    return objects
