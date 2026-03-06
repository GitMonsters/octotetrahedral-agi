#!/usr/bin/env python3
"""
Object-reasoning ARC solver — detects objects, finds relationships,
matches transforms across training examples, applies to test.

This targets the ~70% of unsolved tasks that are same-size transforms
involving object manipulation that the DSL solver misses.
"""

import os
import sys
import json
import time
import multiprocessing
import numpy as np
from typing import Optional, List, Tuple, Dict, Set
from collections import Counter, defaultdict
from scipy import ndimage


def grids_match(a, b) -> bool:
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and np.array_equal(a, b)


# ═══════════════════════════════════════════════════════════
# Object Detection
# ═══════════════════════════════════════════════════════════

def detect_objects(grid: np.ndarray, bg=0) -> List[dict]:
    """Detect connected components as objects."""
    objects = []
    mask = grid != bg
    if not mask.any():
        return objects
    
    labeled, n = ndimage.label(mask)
    for i in range(1, n + 1):
        obj_mask = labeled == i
        rows, cols = np.where(obj_mask)
        r1, c1, r2, c2 = rows.min(), cols.min(), rows.max() + 1, cols.max() + 1
        sprite = grid[r1:r2, c1:c2].copy()
        sprite_mask = obj_mask[r1:r2, c1:c2]
        sprite[~sprite_mask] = bg
        
        colors = set(grid[obj_mask].flatten()) - {bg}
        objects.append({
            'id': i,
            'bbox': (r1, c1, r2, c2),
            'mask': obj_mask,
            'sprite': sprite,
            'sprite_mask': sprite_mask,
            'size': int(obj_mask.sum()),
            'colors': colors,
            'primary_color': int(Counter(grid[obj_mask].flatten()).most_common(1)[0][0]),
            'center': (float(rows.mean()), float(cols.mean())),
            'width': c2 - c1,
            'height': r2 - r1,
        })
    return objects


def detect_objects_per_color(grid: np.ndarray, bg=0) -> List[dict]:
    """Detect objects by color — each color forms separate objects."""
    objects = []
    for color in set(grid.flatten()) - {bg}:
        mask = grid == color
        labeled, n = ndimage.label(mask)
        for i in range(1, n + 1):
            obj_mask = labeled == i
            rows, cols = np.where(obj_mask)
            r1, c1, r2, c2 = rows.min(), cols.min(), rows.max() + 1, cols.max() + 1
            objects.append({
                'bbox': (r1, c1, r2, c2),
                'mask': obj_mask,
                'size': int(obj_mask.sum()),
                'color': int(color),
                'center': (float(rows.mean()), float(cols.mean())),
                'width': c2 - c1,
                'height': r2 - r1,
            })
    return objects


# ═══════════════════════════════════════════════════════════
# Transform Library — things you can do to grids/objects
# ═══════════════════════════════════════════════════════════

def find_background(grid: np.ndarray) -> int:
    """Most common value = background."""
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def recolor_objects(grid: np.ndarray, color_map: dict, bg=0) -> np.ndarray:
    """Recolor each object based on some property → color mapping."""
    result = grid.copy()
    for old_c, new_c in color_map.items():
        result[grid == old_c] = new_c
    return result


def fill_enclosed_regions(grid: np.ndarray, fill_color: int, bg=0) -> np.ndarray:
    """Fill regions enclosed by non-background cells."""
    result = grid.copy()
    h, w = grid.shape
    
    # Flood fill from edges to find "outside" bg cells
    outside = np.zeros_like(grid, dtype=bool)
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if grid[r, c] == bg:
                queue.append((r, c))
                outside[r, c] = True
    for c in range(w):
        for r in [0, h-1]:
            if grid[r, c] == bg and not outside[r, c]:
                queue.append((r, c))
                outside[r, c] = True
    
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not outside[nr, nc] and grid[nr, nc] == bg:
                outside[nr, nc] = True
                queue.append((nr, nc))
    
    # Fill interior bg cells
    interior = (grid == bg) & ~outside
    result[interior] = fill_color
    return result


def remove_color(grid: np.ndarray, color: int, bg=0) -> np.ndarray:
    """Remove all cells of a specific color (set to bg)."""
    result = grid.copy()
    result[grid == color] = bg
    return result


def keep_largest_object(grid: np.ndarray, bg=0) -> np.ndarray:
    """Keep only the largest connected component."""
    objects = detect_objects(grid, bg)
    if not objects:
        return grid.copy()
    largest = max(objects, key=lambda o: o['size'])
    result = np.full_like(grid, bg)
    result[largest['mask']] = grid[largest['mask']]
    return result


def keep_smallest_object(grid: np.ndarray, bg=0) -> np.ndarray:
    """Keep only the smallest connected component."""
    objects = detect_objects(grid, bg)
    if not objects:
        return grid.copy()
    smallest = min(objects, key=lambda o: o['size'])
    result = np.full_like(grid, bg)
    result[smallest['mask']] = grid[smallest['mask']]
    return result


def overlay_objects(grid: np.ndarray, bg=0) -> np.ndarray:
    """Stack/overlay all objects onto smallest bounding box."""
    objects = detect_objects(grid, bg)
    if len(objects) < 2:
        return grid.copy()
    
    # Find common sprite size
    sizes = [(o['height'], o['width']) for o in objects]
    if len(set(sizes)) != 1:
        return grid.copy()
    
    h, w = sizes[0]
    result = np.full((h, w), bg, dtype=grid.dtype)
    for obj in objects:
        r1, c1, r2, c2 = obj['bbox']
        sprite = obj['sprite']
        mask = obj['sprite_mask']
        result[mask] = sprite[mask]
    return result


def xor_objects(grid: np.ndarray, bg=0) -> np.ndarray:
    """XOR of two same-size objects — keep cells that differ."""
    objects = detect_objects(grid, bg)
    if len(objects) != 2:
        return grid.copy()
    
    o1, o2 = objects[0], objects[1]
    if (o1['height'], o1['width']) != (o2['height'], o2['width']):
        return grid.copy()
    
    s1, s2 = o1['sprite'], o2['sprite']
    m1, m2 = o1['sprite_mask'], o2['sprite_mask']
    
    h, w = s1.shape
    result = np.full((h, w), bg, dtype=grid.dtype)
    for r in range(h):
        for c in range(w):
            if m1[r,c] and not m2[r,c]:
                result[r,c] = s1[r,c]
            elif m2[r,c] and not m1[r,c]:
                result[r,c] = s2[r,c]
    return result


def and_objects(grid: np.ndarray, bg=0) -> np.ndarray:
    """AND of two same-size objects — keep cells that overlap."""
    objects = detect_objects(grid, bg)
    if len(objects) != 2:
        return grid.copy()
    
    o1, o2 = objects[0], objects[1]
    if (o1['height'], o1['width']) != (o2['height'], o2['width']):
        return grid.copy()
    
    s1, s2 = o1['sprite'], o2['sprite']
    m1, m2 = o1['sprite_mask'], o2['sprite_mask']
    
    h, w = s1.shape
    result = np.full((h, w), bg, dtype=grid.dtype)
    for r in range(h):
        for c in range(w):
            if m1[r,c] and m2[r,c]:
                result[r,c] = s1[r,c]
    return result


def mirror_along_axis(grid: np.ndarray, axis_color: int, bg=0) -> np.ndarray:
    """Mirror non-background cells across a line of specific color."""
    result = grid.copy()
    h, w = grid.shape
    
    # Find horizontal axis
    for r in range(h):
        if all(grid[r, c] == axis_color or grid[r, c] == bg for c in range(w)) and np.sum(grid[r] == axis_color) > w // 2:
            # Mirror across row r
            for dr in range(1, h):
                if r - dr >= 0 and r + dr < h:
                    for c in range(w):
                        if grid[r-dr, c] != bg and grid[r+dr, c] == bg:
                            result[r+dr, c] = grid[r-dr, c]
                        elif grid[r+dr, c] != bg and grid[r-dr, c] == bg:
                            result[r-dr, c] = grid[r+dr, c]
            return result
    
    # Find vertical axis
    for c in range(w):
        if all(grid[r, c] == axis_color or grid[r, c] == bg for r in range(h)) and np.sum(grid[:, c] == axis_color) > h // 2:
            for dc in range(1, w):
                if c - dc >= 0 and c + dc < w:
                    for r in range(h):
                        if grid[r, c-dc] != bg and grid[r, c+dc] == bg:
                            result[r, c+dc] = grid[r, c-dc]
                        elif grid[r, c+dc] != bg and grid[r, c-dc] == bg:
                            result[r, c-dc] = grid[r, c+dc]
            return result
    
    return grid.copy()


def complete_pattern(grid: np.ndarray, bg=0) -> np.ndarray:
    """Complete a partially drawn pattern by finding symmetry."""
    result = grid.copy()
    h, w = grid.shape
    
    # Try 4-fold symmetry
    for r in range(h):
        for c in range(w):
            candidates = [
                grid[r, c],
                grid[r, w-1-c] if w-1-c != c else bg,
                grid[h-1-r, c] if h-1-r != r else bg,
                grid[h-1-r, w-1-c] if h-1-r != r and w-1-c != c else bg,
            ]
            non_bg = [v for v in candidates if v != bg]
            if non_bg and result[r, c] == bg:
                result[r, c] = Counter(non_bg).most_common(1)[0][0]
    
    return result


def flood_fill_colors(grid: np.ndarray, bg=0) -> np.ndarray:
    """For each enclosed region, fill with the color of its boundary."""
    result = grid.copy()
    h, w = grid.shape
    
    # Find all bg regions
    visited = np.zeros_like(grid, dtype=bool)
    
    for start_r in range(h):
        for start_c in range(w):
            if visited[start_r, start_c] or grid[start_r, start_c] != bg:
                continue
            
            # BFS to find this bg region
            region = []
            border_colors = []
            queue = [(start_r, start_c)]
            visited[start_r, start_c] = True
            touches_edge = False
            
            while queue:
                r, c = queue.pop(0)
                region.append((r, c))
                if r == 0 or r == h-1 or c == 0 or c == w-1:
                    touches_edge = True
                
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not visited[nr, nc]:
                            if grid[nr, nc] == bg:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                            else:
                                border_colors.append(int(grid[nr, nc]))
                    else:
                        touches_edge = True
            
            # Fill enclosed regions
            if not touches_edge and border_colors:
                fill = Counter(border_colors).most_common(1)[0][0]
                for r, c in region:
                    result[r, c] = fill
    
    return result


def gravity_drop(grid: np.ndarray, direction='down', bg=0) -> np.ndarray:
    """Drop all non-bg cells in a direction."""
    result = np.full_like(grid, bg)
    h, w = grid.shape
    
    if direction == 'down':
        for c in range(w):
            vals = [grid[r, c] for r in range(h) if grid[r, c] != bg]
            for i, v in enumerate(reversed(vals)):
                result[h-1-i, c] = v
    elif direction == 'up':
        for c in range(w):
            vals = [grid[r, c] for r in range(h) if grid[r, c] != bg]
            for i, v in enumerate(vals):
                result[i, c] = v
    elif direction == 'right':
        for r in range(h):
            vals = [grid[r, c] for c in range(w) if grid[r, c] != bg]
            for i, v in enumerate(reversed(vals)):
                result[r, w-1-i] = v
    elif direction == 'left':
        for r in range(h):
            vals = [grid[r, c] for c in range(w) if grid[r, c] != bg]
            for i, v in enumerate(vals):
                result[r, i] = v
    
    return result


def crop_to_content(grid: np.ndarray, bg=0) -> np.ndarray:
    """Crop grid to bounding box of non-bg content."""
    rows, cols = np.where(grid != bg)
    if len(rows) == 0:
        return grid.copy()
    return grid[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()


def extract_repeated_pattern(grid: np.ndarray, bg=0) -> np.ndarray:
    """Find the smallest repeating tile in the grid."""
    h, w = grid.shape
    for th in range(1, h+1):
        if h % th != 0:
            continue
        for tw in range(1, w+1):
            if w % tw != 0:
                continue
            tile = grid[:th, :tw]
            tiled = np.tile(tile, (h//th, w//tw))
            if np.array_equal(tiled, grid):
                return tile
    return grid.copy()


def scale_up(grid: np.ndarray, factor: int) -> np.ndarray:
    """Scale grid up by integer factor."""
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


def scale_down(grid: np.ndarray, factor: int, bg=0) -> np.ndarray:
    """Scale grid down by integer factor (mode aggregation)."""
    h, w = grid.shape
    if h % factor != 0 or w % factor != 0:
        return grid.copy()
    oh, ow = h // factor, w // factor
    result = np.zeros((oh, ow), dtype=grid.dtype)
    for r in range(oh):
        for c in range(ow):
            block = grid[r*factor:(r+1)*factor, c*factor:(c+1)*factor]
            vals = block.flatten()
            non_bg = vals[vals != bg]
            if len(non_bg) > 0:
                result[r, c] = Counter(non_bg.tolist()).most_common(1)[0][0]
            else:
                result[r, c] = bg
    return result


# ═══════════════════════════════════════════════════════════
# Meta-solver: try all transforms and compositions
# ═══════════════════════════════════════════════════════════

def get_all_transforms(grid: np.ndarray, bg=0) -> List[Tuple[str, np.ndarray]]:
    """Generate all candidate transforms of a grid."""
    h, w = grid.shape
    candidates = []
    
    # Identity
    candidates.append(('identity', grid.copy()))
    
    # Geometric
    candidates.append(('fliplr', np.fliplr(grid)))
    candidates.append(('flipud', np.flipud(grid)))
    candidates.append(('rot90', np.rot90(grid, 1)))
    candidates.append(('rot180', np.rot90(grid, 2)))
    candidates.append(('rot270', np.rot90(grid, 3)))
    candidates.append(('transpose', grid.T))
    
    # Color operations
    colors = set(grid.flatten().astype(int)) - {bg}
    for color in colors:
        candidates.append((f'remove_{color}', remove_color(grid, color, bg)))
        candidates.append((f'fill_enclosed_{color}', fill_enclosed_regions(grid, color, bg)))
    
    # Fill operations
    candidates.append(('fill_enclosed_border', flood_fill_colors(grid, bg)))
    candidates.append(('complete_symmetry', complete_pattern(grid, bg)))
    
    # Object operations
    candidates.append(('keep_largest', keep_largest_object(grid, bg)))
    candidates.append(('keep_smallest', keep_smallest_object(grid, bg)))
    
    # Gravity
    for d in ['down', 'up', 'left', 'right']:
        candidates.append((f'gravity_{d}', gravity_drop(grid, d, bg)))
    
    # Mirror across each color axis
    for color in colors:
        candidates.append((f'mirror_{color}', mirror_along_axis(grid, color, bg)))
    
    # Crop
    candidates.append(('crop', crop_to_content(grid, bg)))
    
    # Scale
    for f in [2, 3]:
        candidates.append((f'scale_up_{f}', scale_up(grid, f)))
        candidates.append((f'scale_down_{f}', scale_down(grid, f, bg)))
    
    # Tiling
    candidates.append(('tile_2x2', np.tile(grid, (2, 2))))
    candidates.append(('tile_3x3', np.tile(grid, (3, 3))))
    candidates.append(('tile_1x2', np.tile(grid, (1, 2))))
    candidates.append(('tile_2x1', np.tile(grid, (2, 1))))
    
    # Extract repeating pattern
    candidates.append(('extract_repeat', extract_repeated_pattern(grid, bg)))
    
    # Overlay / boolean ops on objects
    candidates.append(('overlay', overlay_objects(grid, bg)))
    candidates.append(('xor_objects', xor_objects(grid, bg)))
    candidates.append(('and_objects', and_objects(grid, bg)))
    
    return candidates


def solve_by_transform_matching(task: dict) -> Optional[Tuple[list, str]]:
    """Try single transforms and depth-2 compositions."""
    train = task['train']
    test_input = np.array(task['test'][0]['input'])
    bg = find_background(test_input)
    
    # Phase 1: single transforms
    for name, pred in get_all_transforms(np.array(train[0]['input']), bg):
        expected = np.array(train[0]['output'])
        if pred.shape != expected.shape or not np.array_equal(pred, expected):
            continue
        
        # Verify on all training examples
        all_match = True
        for ex in train[1:]:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            for n2, p2 in get_all_transforms(inp, find_background(inp)):
                if n2 == name and p2.shape == out.shape and np.array_equal(p2, out):
                    break
            else:
                all_match = False
                break
        
        if all_match:
            # Apply to test
            for n3, p3 in get_all_transforms(test_input, bg):
                if n3 == name:
                    return p3.tolist(), name
    
    # Phase 2: depth-2 compositions (limited to avoid explosion)
    fast_transforms = [
        ('fliplr', lambda g, bg: np.fliplr(g)),
        ('flipud', lambda g, bg: np.flipud(g)),
        ('rot90', lambda g, bg: np.rot90(g, 1)),
        ('rot180', lambda g, bg: np.rot90(g, 2)),
        ('transpose', lambda g, bg: g.T),
        ('fill_enclosed', lambda g, bg: flood_fill_colors(g, bg)),
        ('complete_sym', lambda g, bg: complete_pattern(g, bg)),
        ('keep_largest', lambda g, bg: keep_largest_object(g, bg)),
        ('keep_smallest', lambda g, bg: keep_smallest_object(g, bg)),
        ('gravity_down', lambda g, bg: gravity_drop(g, 'down', bg)),
        ('gravity_up', lambda g, bg: gravity_drop(g, 'up', bg)),
        ('crop', lambda g, bg: crop_to_content(g, bg)),
    ]
    
    for n1, fn1 in fast_transforms:
        for n2, fn2 in fast_transforms:
            if n1 == n2 and n1 in ('fliplr', 'flipud', 'rot180'):
                continue  # self-inverse, skip
            
            all_match = True
            for ex in train:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                bg_i = find_background(inp)
                try:
                    step1 = fn1(inp, bg_i)
                    step2 = fn2(step1, bg_i)
                    if step2.shape != out.shape or not np.array_equal(step2, out):
                        all_match = False
                        break
                except Exception:
                    all_match = False
                    break
            
            if all_match:
                try:
                    step1 = fn1(test_input, bg)
                    step2 = fn2(step1, bg)
                    return step2.tolist(), f'{n1}+{n2}'
                except Exception:
                    pass
    
    return None


def solve_by_object_reasoning(task: dict) -> Optional[Tuple[list, str]]:
    """Solve by understanding objects and their relationships."""
    train = task['train']
    
    # Strategy: find what happens to objects between input and output
    for ex in train:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        if inp.shape != out.shape:
            break
    else:
        # All same-size — try object-level analysis
        # Check: does each object get individually transformed?
        pass
    
    return None


def solve_by_counting(task: dict) -> Optional[Tuple[list, str]]:
    """Solve tasks where output encodes counts or statistics."""
    train = task['train']
    test_input = np.array(task['test'][0]['input'])
    
    # Check: is output a single cell (1x1)?
    out0 = np.array(train[0]['output'])
    if out0.size == 1:
        # Output is a count/statistic of input
        for ex in train:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            if out.size != 1:
                return None
        
        # Try various counting rules
        rules = [
            ('n_colors', lambda g: len(set(g.flatten()) - {0})),
            ('n_objects', lambda g: len(detect_objects(g, find_background(g)))),
            ('max_color', lambda g: int(g.max())),
            ('min_nonzero', lambda g: int(g[g > 0].min()) if g[g > 0].size > 0 else 0),
            ('n_nonzero', lambda g: int(np.count_nonzero(g))),
        ]
        
        for rule_name, rule_fn in rules:
            all_match = True
            for ex in train:
                inp = np.array(ex['input'])
                out_val = int(np.array(ex['output']).flatten()[0])
                try:
                    pred_val = rule_fn(inp)
                    if pred_val != out_val:
                        all_match = False
                        break
                except Exception:
                    all_match = False
                    break
            
            if all_match:
                val = rule_fn(test_input)
                return [[val]], f'count:{rule_name}'
    
    # Check: is output a small grid encoding statistics?
    # e.g., 1xN grid where each cell = count of color N
    if out0.shape[0] == 1 or out0.shape[1] == 1:
        n = max(out0.shape)
        # Try: output[i] = count of color i in input
        all_match = True
        for ex in train:
            inp = np.array(ex['input'])
            out = np.array(ex['output']).flatten()
            if len(out) > 10:
                all_match = False
                break
            for i in range(len(out)):
                count = int(np.sum(inp == (i + 1)))
                if count != out[i]:
                    all_match = False
                    break
            if not all_match:
                break
        
        if all_match:
            n = len(np.array(train[0]['output']).flatten())
            result = [int(np.sum(test_input == (i + 1))) for i in range(n)]
            if out0.shape[0] == 1:
                return [result], 'count_colors_row'
            else:
                return [[v] for v in result], 'count_colors_col'
    
    return None


# ═══════════════════════════════════════════════════════════
# Main solver
# ═══════════════════════════════════════════════════════════

def solve_task(task: dict) -> Optional[Tuple[list, str]]:
    """Try all solving strategies."""
    # Fast checks first
    result = solve_by_counting(task)
    if result:
        return result
    
    result = solve_by_transform_matching(task)
    if result:
        return result
    
    return None


def _worker(task_json, result_queue):
    """Subprocess worker."""
    task = json.loads(task_json)
    test_output = task['test'][0].get('output')
    if test_output is None:
        result_queue.put((False, 'no_gt'))
        return
    
    result = solve_task(task)
    if result is not None:
        pred, method = result
        if grids_match(pred, test_output):
            result_queue.put((True, method))
            return
    
    result_queue.put((False, 'none'))


def solve_with_timeout(task_json, timeout_s=30):
    """Run in forked subprocess with hard timeout."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(task_json, q))
    p.start()
    p.join(timeout=timeout_s)
    
    if p.is_alive():
        p.terminate()
        p.join(timeout=3)
        return False, 'timeout'
    
    if not q.empty():
        try:
            return q.get_nowait()
        except Exception:
            pass
    return False, 'error'


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)
    
    eval_dir = 'ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation'
    
    prev_solved = set()
    for f in ['arc_agi1_eval_results.json', 'arc_emerged_eval_results.json']:
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            prev_solved |= set(d.get('solved_ids', []))

    logger.info("=" * 60)
    logger.info("🧩 OBJECT-REASONING SOLVER — ARC-AGI-1 EVAL")
    logger.info(f"Previously solved: {len(prev_solved)}/400")
    logger.info("=" * 60)

    solved = []
    methods = {}
    total = 0
    t_start = time.time()

    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith('.json'):
            continue
        tid = fn.replace('.json', '')
        total += 1

        with open(os.path.join(eval_dir, fn)) as f:
            task = json.load(f)

        t0 = time.time()
        is_solved, method = solve_with_timeout(json.dumps(task), timeout_s=30)
        elapsed = time.time() - t0

        if is_solved:
            solved.append(tid)
            methods[method] = methods.get(method, 0) + 1
            is_new = "🆕" if tid not in prev_solved else ""
            logger.info(f"  ✅ {tid} via {method} ({elapsed:.1f}s) {is_new}")
        
        if total % 50 == 0:
            wall = time.time() - t_start
            eta = wall / total * (400 - total)
            new = len([t for t in solved if t not in prev_solved])
            logger.info(f"  [{total}/400] {len(solved)} solved (+{new} new) | {wall:.0f}s | ETA: {eta/60:.0f}m")
        sys.stdout.flush()

    elapsed = time.time() - t_start
    new_solves = [t for t in solved if t not in prev_solved]
    combined = prev_solved | set(solved)

    logger.info(f"\n{'='*60}")
    logger.info("📊 OBJECT-REASONING RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  This solver:    {len(solved)}/{total}")
    logger.info(f"  New:            {len(new_solves)}")
    logger.info(f"  ★ TOTAL:        {len(combined)}/400 ({len(combined)/4:.1f}%)")
    logger.info(f"  Time:           {elapsed:.0f}s")
    logger.info(f"  Methods:        {methods}")
    if new_solves:
        logger.info(f"  🆕 New IDs:     {sorted(new_solves)}")

    output = {
        'solver_solved': len(solved),
        'new_solves': len(new_solves),
        'total': len(combined),
        'total_pct': round(len(combined) / 4, 1),
        'solved_ids': sorted(solved),
        'new_solve_ids': sorted(new_solves),
        'combined_ids': sorted(combined),
        'methods': methods,
        'time_s': round(elapsed, 1),
    }
    with open('arc_object_reasoning_eval_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Saved to arc_object_reasoning_eval_results.json")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
