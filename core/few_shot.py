"""
Few-Shot Abstraction Engine

Given 2-3 (input, output) example pairs from an ARC task, extracts
a latent "rule" by analyzing structural differences:
  - Grid dimension changes
  - Color mapping patterns
  - Spatial transformations
  - Object-level operations (move, copy, fill, crop)
  - Symmetry and periodicity

The abstracted rule is represented as a structured hypothesis that
can be composed from primitives and tested against held-out examples.
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter, defaultdict
from itertools import combinations
import copy


# ============================================================================
# Grid Analysis Utilities
# ============================================================================

def grid_dims(grid: List[List[int]]) -> Tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0

def grid_colors(grid: List[List[int]]) -> Set[int]:
    return {c for row in grid for c in row}

def color_counts(grid: List[List[int]]) -> Dict[int, int]:
    return Counter(c for row in grid for c in row)

def grid_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    if len(a) != len(b):
        return False
    return all(ra == rb for ra, rb in zip(a, b))

def find_objects(grid: List[List[int]], bg: int = 0) -> List[Dict[str, Any]]:
    """Find connected components (objects) via flood fill."""
    rows, cols = grid_dims(grid)
    visited = [[False] * cols for _ in range(rows)]
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
                        'color': grid[r][c],
                        'cells': cells,
                        'bbox': (min_r, min_c, max_r, max_c),
                        'size': len(cells),
                        'width': max_c - min_c + 1,
                        'height': max_r - min_r + 1,
                    })
    return objects


def extract_subgrid(grid: List[List[int]], bbox: Tuple[int,int,int,int]) -> List[List[int]]:
    r1, c1, r2, c2 = bbox
    return [row[c1:c2+1] for row in grid[r1:r2+1]]


def background_color(grid: List[List[int]]) -> int:
    counts = color_counts(grid)
    return counts.most_common(1)[0][0] if counts else 0


# ============================================================================
# Few-Shot Abstraction
# ============================================================================

class FewShotAbstractor:
    """
    Extracts structural transformation rules from few (input, output) examples.
    
    Returns a ranked list of AbstractedRule objects describing what changes
    between input and output across all examples.
    """

    def abstract(self, examples: List[Dict]) -> List[Dict[str, Any]]:
        """
        Given training examples [{input, output}, ...], extract rules.
        
        Returns list of candidate rules sorted by confidence, each containing:
          - type: str (e.g., 'color_map', 'resize', 'object_move', ...)
          - params: dict of rule parameters
          - confidence: float 0-1
        """
        rules = []

        # 1. Dimension analysis
        dim_rule = self._abstract_dimensions(examples)
        if dim_rule:
            rules.append(dim_rule)

        # 2. Color mapping
        cmap_rule = self._abstract_color_mapping(examples)
        if cmap_rule:
            rules.append(cmap_rule)

        # 3. Geometric transforms
        geo_rules = self._abstract_geometric(examples)
        rules.extend(geo_rules)

        # 4. Object-level analysis
        obj_rules = self._abstract_objects(examples)
        rules.extend(obj_rules)

        # 5. Tiling / periodicity
        tile_rule = self._abstract_tiling(examples)
        if tile_rule:
            rules.append(tile_rule)

        # 6. Symmetry completion
        sym_rule = self._abstract_symmetry(examples)
        if sym_rule:
            rules.append(sym_rule)

        # 7. Border / frame patterns
        border_rule = self._abstract_border(examples)
        if border_rule:
            rules.append(border_rule)

        # 8. Gravity / stacking
        grav_rule = self._abstract_gravity(examples)
        if grav_rule:
            rules.append(grav_rule)

        # Sort by confidence
        rules.sort(key=lambda r: r['confidence'], reverse=True)
        return rules

    def _abstract_dimensions(self, examples: List[Dict]) -> Optional[Dict]:
        """Check if output dimensions relate to input dimensions consistently."""
        ratios = []
        for ex in examples:
            ir, ic = grid_dims(ex['input'])
            or_, oc = grid_dims(ex['output'])
            if ir == 0 or ic == 0:
                return None
            ratios.append((or_ / ir, oc / ic))

        if len(set(ratios)) == 1:
            rr, rc = ratios[0]
            if rr == rc and rr == int(rr) and rr > 1:
                return {
                    'type': 'scale',
                    'params': {'factor': int(rr)},
                    'confidence': 0.95,
                }
            elif rr != 1.0 or rc != 1.0:
                return {
                    'type': 'resize',
                    'params': {'row_ratio': rr, 'col_ratio': rc},
                    'confidence': 0.7,
                }
        # Check if output is always same dims
        out_dims = [grid_dims(ex['output']) for ex in examples]
        if len(set(out_dims)) == 1 and len(set(grid_dims(ex['input']) for ex in examples)) > 1:
            return {
                'type': 'fixed_output_size',
                'params': {'rows': out_dims[0][0], 'cols': out_dims[0][1]},
                'confidence': 0.6,
            }
        return None

    def _abstract_color_mapping(self, examples: List[Dict]) -> Optional[Dict]:
        """Check if there's a consistent color substitution."""
        mappings = []
        for ex in examples:
            ir, ic = grid_dims(ex['input'])
            or_, oc = grid_dims(ex['output'])
            if ir != or_ or ic != oc:
                return None  # dims must match for pixel-wise mapping
            mapping = {}
            valid = True
            for r in range(ir):
                for c in range(ic):
                    ci = ex['input'][r][c]
                    co = ex['output'][r][c]
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

        # Check consistency across examples
        if not mappings:
            return None
        ref = mappings[0]
        for m in mappings[1:]:
            for k, v in ref.items():
                if k in m and m[k] != v:
                    return None

        # Merge all mappings
        merged = {}
        for m in mappings:
            merged.update(m)

        # Skip identity mapping
        if all(k == v for k, v in merged.items()):
            return None

        return {
            'type': 'color_map',
            'params': {'mapping': merged},
            'confidence': 0.95,
        }

    def _abstract_geometric(self, examples: List[Dict]) -> List[Dict]:
        """Try standard geometric transforms."""
        transforms = {
            'rotate_90': lambda g: [list(row) for row in zip(*g[::-1])],
            'rotate_180': lambda g: [row[::-1] for row in g[::-1]],
            'rotate_270': lambda g: [list(row) for row in zip(*[r[::-1] for r in g])],
            'flip_h': lambda g: [row[::-1] for row in g],
            'flip_v': lambda g: g[::-1],
            'transpose': lambda g: [list(row) for row in zip(*g)],
        }
        results = []
        for name, fn in transforms.items():
            if all(grid_equal(fn(ex['input']), ex['output']) for ex in examples):
                results.append({
                    'type': 'geometric',
                    'params': {'transform': name},
                    'confidence': 1.0,
                })
        return results

    def _abstract_objects(self, examples: List[Dict]) -> List[Dict]:
        """Analyze object-level changes between input and output."""
        rules = []
        for ex in examples:
            bg_in = background_color(ex['input'])
            bg_out = background_color(ex['output'])
            objs_in = find_objects(ex['input'], bg_in)
            objs_out = find_objects(ex['output'], bg_out)

            # Check: output = largest object extracted
            if len(objs_in) > 1 and len(objs_out) == 1:
                largest = max(objs_in, key=lambda o: o['size'])
                extracted = extract_subgrid(ex['input'], largest['bbox'])
                if grid_equal(extracted, ex['output']):
                    rules.append({
                        'type': 'extract_largest_object',
                        'params': {},
                        'confidence': 0.85,
                    })
                    break

            # Check: output = smallest object extracted
            if len(objs_in) > 1 and len(objs_out) == 1:
                smallest = min(objs_in, key=lambda o: o['size'])
                extracted = extract_subgrid(ex['input'], smallest['bbox'])
                if grid_equal(extracted, ex['output']):
                    rules.append({
                        'type': 'extract_smallest_object',
                        'params': {},
                        'confidence': 0.8,
                    })
                    break

            # Check: count objects → output encodes count
            if len(objs_in) >= 1:
                or_, oc = grid_dims(ex['output'])
                if or_ == 1 and oc == 1:
                    if ex['output'][0][0] == len(objs_in):
                        rules.append({
                            'type': 'count_objects',
                            'params': {},
                            'confidence': 0.75,
                        })
                        break

        return rules

    def _abstract_tiling(self, examples: List[Dict]) -> Optional[Dict]:
        """Check if output is input tiled N×M times."""
        for h in range(1, 5):
            for w in range(1, 5):
                if h == 1 and w == 1:
                    continue
                match = True
                for ex in examples:
                    ir, ic = grid_dims(ex['input'])
                    or_, oc = grid_dims(ex['output'])
                    if or_ != ir * h or oc != ic * w:
                        match = False
                        break
                    # Verify tiling content
                    for tr in range(h):
                        for tc in range(w):
                            for r in range(ir):
                                for c in range(ic):
                                    if ex['output'][tr*ir+r][tc*ic+c] != ex['input'][r][c]:
                                        match = False
                                        break
                                if not match:
                                    break
                            if not match:
                                break
                        if not match:
                            break
                if match:
                    return {
                        'type': 'tile',
                        'params': {'rows': h, 'cols': w},
                        'confidence': 0.95,
                    }
        return None

    def _abstract_symmetry(self, examples: List[Dict]) -> Optional[Dict]:
        """Check if output completes a symmetry in input."""
        for ex in examples:
            ir, ic = grid_dims(ex['input'])
            or_, oc = grid_dims(ex['output'])
            if ir != or_ or ic != oc:
                continue
            # Check vertical symmetry completion
            bg = background_color(ex['input'])
            sym_v = True
            for r in range(ir):
                for c in range(ic):
                    mirror_c = ic - 1 - c
                    if ex['input'][r][c] != bg and ex['input'][r][mirror_c] == bg:
                        if ex['output'][r][mirror_c] != ex['input'][r][c]:
                            sym_v = False
                            break
                if not sym_v:
                    break
            if sym_v:
                return {
                    'type': 'symmetry_completion',
                    'params': {'axis': 'vertical'},
                    'confidence': 0.7,
                }
        return None

    def _abstract_border(self, examples: List[Dict]) -> Optional[Dict]:
        """Check if a border/frame is added or removed."""
        for ex in examples:
            ir, ic = grid_dims(ex['input'])
            or_, oc = grid_dims(ex['output'])
            # Border added (output is 2 larger in each dimension)
            if or_ == ir + 2 and oc == ic + 2:
                # Check if inner content matches
                inner = extract_subgrid(ex['output'], (1, 1, or_-2, oc-2))
                if grid_equal(inner, ex['input']):
                    border_color = ex['output'][0][0]
                    return {
                        'type': 'add_border',
                        'params': {'color': border_color, 'width': 1},
                        'confidence': 0.85,
                    }
            # Border removed
            if or_ == ir - 2 and oc == ic - 2 and ir > 2 and ic > 2:
                inner = extract_subgrid(ex['input'], (1, 1, ir-2, ic-2))
                if grid_equal(inner, ex['output']):
                    return {
                        'type': 'remove_border',
                        'params': {'width': 1},
                        'confidence': 0.85,
                    }
        return None

    def _abstract_gravity(self, examples: List[Dict]) -> Optional[Dict]:
        """Check if non-background cells fall to bottom."""
        for direction in ['down', 'up', 'left', 'right']:
            match = True
            for ex in examples:
                ir, ic = grid_dims(ex['input'])
                or_, oc = grid_dims(ex['output'])
                if ir != or_ or ic != oc:
                    match = False
                    break
                expected = self._apply_gravity(ex['input'], direction)
                if not grid_equal(expected, ex['output']):
                    match = False
                    break
            if match:
                return {
                    'type': 'gravity',
                    'params': {'direction': direction},
                    'confidence': 0.9,
                }
        return None

    def _apply_gravity(self, grid: List[List[int]], direction: str) -> List[List[int]]:
        rows, cols = grid_dims(grid)
        bg = background_color(grid)
        result = [[bg] * cols for _ in range(rows)]
        if direction == 'down':
            for c in range(cols):
                non_bg = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
                for i, v in enumerate(non_bg):
                    result[rows - len(non_bg) + i][c] = v
        elif direction == 'up':
            for c in range(cols):
                non_bg = [grid[r][c] for r in range(rows) if grid[r][c] != bg]
                for i, v in enumerate(non_bg):
                    result[i][c] = v
        elif direction == 'left':
            for r in range(rows):
                non_bg = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
                for i, v in enumerate(non_bg):
                    result[r][i] = v
        elif direction == 'right':
            for r in range(rows):
                non_bg = [grid[r][c] for c in range(cols) if grid[r][c] != bg]
                for i, v in enumerate(non_bg):
                    result[r][cols - len(non_bg) + i] = v
        return result
