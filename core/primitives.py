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
        
        Returns list of solutions, each containing:
          - program: list of primitive names
          - depth: number of chained primitives
        """
        depth = max_depth or self.max_depth
        solutions = []

        # Depth 1: single primitives
        for name, fn in self.primitives:
            if self._test_program([fn], examples):
                solutions.append({'program': [name], 'depth': 1})

        if solutions or depth < 2:
            return solutions

        # Depth 2: pairs
        for n1, f1 in self.primitives:
            for n2, f2 in self.primitives:
                if n1 == 'identity' or n2 == 'identity':
                    continue
                if self._test_program([f1, f2], examples):
                    solutions.append({'program': [n1, n2], 'depth': 2})

        if solutions or depth < 3:
            return solutions

        # Depth 3: triples (expensive — prune aggressively)
        # Only try if depth-2 partial matches exist
        partial_first = []
        for n1, f1 in self.primitives:
            if n1 == 'identity':
                continue
            partial_ok = False
            for ex in examples[:1]:  # Quick check on first example only
                try:
                    r = f1(ex['input'])
                    if r != ex['input']:  # Actually does something
                        partial_ok = True
                except Exception:
                    pass
            if partial_ok:
                partial_first.append((n1, f1))

        for n1, f1 in partial_first:
            for n2, f2 in self.primitives:
                if n2 == 'identity':
                    continue
                for n3, f3 in self.primitives:
                    if n3 == 'identity':
                        continue
                    if self._test_program([f1, f2, f3], examples):
                        solutions.append({'program': [n1, n2, n3], 'depth': 3})

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
