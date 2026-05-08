#!/usr/bin/env python3
"""
RE-ARC v50: Per-Task Rule Learner

Strategy: Learn the transformation rule from training pairs, verify across ALL
pairs before applying. Far more accurate than catalog ensemble on unseen tasks.

Rules tried (in order):
1. Geometric: rot90, fliplr, flipud, transpose (all variants)
2. Global color permutation (consistent map across all train pairs)
3. Background elimination (non-bg → bg)
4. Scale-up (each pixel → NxN block)
5. Scale-down (detect zoom factor)
6. Tiling detection
7. Fallback: identity
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import Counter

sys.path.insert(0, '/Users/evanpieser')


def _to_list(arr: np.ndarray) -> List:
    return arr.tolist()


def _all_pairs_match(transform: Callable, train: List[Dict]) -> bool:
    """Return True if transform correctly predicts output for every training pair."""
    for pair in train:
        inp = np.array(pair['input'])
        exp = np.array(pair['output'])
        try:
            pred = transform(inp)
            if not np.array_equal(pred, exp):
                return False
        except Exception:
            return False
    return True


# ============================================================================
# RULE DETECTORS
# ============================================================================

def detect_geometric(train: List[Dict]) -> Optional[Callable]:
    """Detect rotation/flip/transpose consistent across all training pairs."""
    transforms = [
        ('rot90_1',   lambda x: np.rot90(x, 1)),
        ('rot90_2',   lambda x: np.rot90(x, 2)),
        ('rot90_3',   lambda x: np.rot90(x, 3)),
        ('fliplr',    lambda x: np.fliplr(x)),
        ('flipud',    lambda x: np.flipud(x)),
        ('transpose', lambda x: x.T),
        ('fliplr+rot90', lambda x: np.rot90(np.fliplr(x))),
        ('flipud+rot90', lambda x: np.rot90(np.flipud(x))),
    ]
    for name, fn in transforms:
        if _all_pairs_match(fn, train):
            return fn
    return None


def detect_color_map(train: List[Dict]) -> Optional[Callable]:
    """
    Detect a consistent global color permutation.
    Same input color must always map to same output color in every pair.
    """
    global_map: Dict[int, int] = {}

    for pair in train:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        if inp.shape != out.shape:
            return None
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                ci, co = int(inp[r, c]), int(out[r, c])
                if ci in global_map:
                    if global_map[ci] != co:
                        return None  # Inconsistent
                else:
                    global_map[ci] = co

    if not global_map:
        return None
    # Require at least one actual change
    if all(k == v for k, v in global_map.items()):
        return None

    def apply_color_map(x: np.ndarray, cmap: Dict[int, int] = global_map) -> np.ndarray:
        result = x.copy()
        for src, dst in cmap.items():
            result[x == src] = dst
        return result

    # Verify on all pairs (should always pass given construction, but double-check)
    if _all_pairs_match(apply_color_map, train):
        return apply_color_map
    return None


def _detect_bg(grid: np.ndarray) -> int:
    """Most frequent color = background."""
    return int(Counter(grid.flatten().tolist()).most_common(1)[0][0])


def detect_bg_operations(train: List[Dict]) -> Optional[Callable]:
    """Detect operations on background vs foreground objects."""
    # Rule: all non-background → background (erase objects)
    def erase_objects(x: np.ndarray) -> np.ndarray:
        bg = _detect_bg(x)
        return np.full(x.shape, bg, dtype=x.dtype)

    if _all_pairs_match(erase_objects, train):
        return erase_objects

    # Rule: background → 0, keep foreground (normalize bg to 0)
    def normalize_bg(x: np.ndarray) -> np.ndarray:
        bg = _detect_bg(x)
        result = x.copy()
        result[x == bg] = 0
        return result

    if _all_pairs_match(normalize_bg, train):
        return normalize_bg

    return None


def detect_scale_up(train: List[Dict]) -> Optional[Callable]:
    """Detect integer upscaling (each pixel → k×k block)."""
    for pair in train[:1]:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh < ih or ow < iw:
            continue
        if oh % ih != 0 or ow % iw != 0:
            continue
        sh, sw = oh // ih, ow // iw
        if sh != sw:
            continue
        k = sh

        def scale(x: np.ndarray, factor: int = k) -> np.ndarray:
            return np.repeat(np.repeat(x, factor, axis=0), factor, axis=1)

        if _all_pairs_match(scale, train):
            return scale
    return None


def detect_scale_down(train: List[Dict]) -> Optional[Callable]:
    """Detect integer downscaling by sampling top-left of each block."""
    for pair in train[:1]:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh > ih or ow > iw:
            continue
        if oh == 0 or ow == 0:
            continue
        if ih % oh != 0 or iw % ow != 0:
            continue
        sh, sw = ih // oh, iw // ow

        def downsample(x: np.ndarray, sh_: int = sh, sw_: int = sw) -> np.ndarray:
            return x[::sh_, ::sw_]

        if _all_pairs_match(downsample, train):
            return downsample

        # Try modal downsampling (most common color in each block)
        def modal_down(x: np.ndarray, sh_: int = sh, sw_: int = sw,
                       oh_: int = oh, ow_: int = ow) -> np.ndarray:
            result = np.zeros((oh_, ow_), dtype=x.dtype)
            for r in range(oh_):
                for c in range(ow_):
                    block = x[r*sh_:(r+1)*sh_, c*sw_:(c+1)*sw_]
                    result[r, c] = int(Counter(block.flatten().tolist()).most_common(1)[0][0])
            return result

        if _all_pairs_match(modal_down, train):
            return modal_down
    return None


def detect_tiling(train: List[Dict]) -> Optional[Callable]:
    """Detect output = input tiled to fill larger output."""
    for pair in train[:1]:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh < ih or ow < iw:
            continue
        if oh % ih != 0 or ow % iw != 0:
            continue
        reps_h, reps_w = oh // ih, ow // iw

        def tile(x: np.ndarray, rh: int = reps_h, rw: int = reps_w) -> np.ndarray:
            return np.tile(x, (rh, rw))

        if _all_pairs_match(tile, train):
            return tile
    return None


def detect_largest_solid_rect(train: List[Dict]) -> Optional[Callable]:
    """
    Find the largest axis-aligned solid rectangle of a single non-bg color.
    Everything else → background. Threshold: area must be > 1.
    """
    def find_lsr(x: np.ndarray) -> np.ndarray:
        bg = _detect_bg(x)
        result = np.full_like(x, bg)
        best_area = 1  # must exceed 1 to suppress isolated pixels
        best_r1, best_r2, best_c1, best_c2, best_color = 0, 0, 0, 0, None

        for color in set(x.flatten().tolist()):
            if color == bg:
                continue
            mask = (x == color)
            h, w = x.shape
            heights = np.zeros(w, dtype=int)
            for r in range(h):
                heights = np.where(mask[r], heights + 1, 0)
                stack: list = []
                for c in range(w + 1):
                    curr_h = int(heights[c]) if c < w else 0
                    start = c
                    while stack and stack[-1][1] > curr_h:
                        sc, sh_ = stack.pop()
                        area = sh_ * (c - sc)
                        if area > best_area:
                            best_area = area
                            best_r1 = r - sh_ + 1
                            best_r2 = r
                            best_c1 = sc
                            best_c2 = c - 1
                            best_color = color
                        start = sc
                    stack.append((start, curr_h))

        if best_color is not None:
            result[best_r1:best_r2 + 1, best_c1:best_c2 + 1] = best_color
        return result

    if _all_pairs_match(find_lsr, train):
        return find_lsr
    return None


def detect_keep_multi_pixel_objects(train: List[Dict]) -> Optional[Callable]:
    """
    Remove isolated single pixels (noise); keep only connected components
    with more than 1 pixel.
    """
    from scipy import ndimage as _ndimage

    def keep_multi(x: np.ndarray) -> np.ndarray:
        bg = _detect_bg(x)
        result = np.full_like(x, bg)
        labeled, num = _ndimage.label(x != bg)
        for lbl in range(1, num + 1):
            region = labeled == lbl
            if region.sum() > 1:
                result[region] = x[region][0]
        return result

    if _all_pairs_match(keep_multi, train):
        return keep_multi
    return None


def detect_extract_object(train: List[Dict]) -> Optional[Callable]:
    """
    Detect: extract bounding box of non-background content.
    Common when output is smaller and contains just the foreground object.
    """
    for pair in train[:1]:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        if out.size >= inp.size:
            continue

        def extract_bbox(x: np.ndarray) -> np.ndarray:
            bg = _detect_bg(x)
            mask = x != bg
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return x
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            return x[rmin:rmax+1, cmin:cmax+1]

        if _all_pairs_match(extract_bbox, train):
            return extract_bbox
    return None


def detect_gravity(train: List[Dict]) -> Optional[Callable]:
    """Detect: non-bg objects fall to bottom/top/left/right."""
    directions = []

    def gravity_down(x: np.ndarray) -> np.ndarray:
        bg = _detect_bg(x)
        result = np.full_like(x, bg)
        for c in range(x.shape[1]):
            col = x[:, c]
            objs = col[col != bg]
            result[x.shape[0]-len(objs):, c] = objs
        return result

    def gravity_up(x: np.ndarray) -> np.ndarray:
        bg = _detect_bg(x)
        result = np.full_like(x, bg)
        for c in range(x.shape[1]):
            col = x[:, c]
            objs = col[col != bg]
            result[:len(objs), c] = objs
        return result

    def gravity_right(x: np.ndarray) -> np.ndarray:
        bg = _detect_bg(x)
        result = np.full_like(x, bg)
        for r in range(x.shape[0]):
            row = x[r, :]
            objs = row[row != bg]
            result[r, x.shape[1]-len(objs):] = objs
        return result

    def gravity_left(x: np.ndarray) -> np.ndarray:
        bg = _detect_bg(x)
        result = np.full_like(x, bg)
        for r in range(x.shape[0]):
            row = x[r, :]
            objs = row[row != bg]
            result[r, :len(objs)] = objs
        return result

    for fn in [gravity_down, gravity_up, gravity_left, gravity_right]:
        if _all_pairs_match(fn, train):
            return fn
    return None


def detect_sort_rows_by_color(train: List[Dict]) -> Optional[Callable]:
    """Detect sorting of rows/columns by dominant color."""

    def sort_rows_asc(x: np.ndarray) -> np.ndarray:
        key = [int(Counter(row.tolist()).most_common(1)[0][0]) for row in x]
        order = np.argsort(key, kind='stable')
        return x[order]

    def sort_rows_desc(x: np.ndarray) -> np.ndarray:
        key = [int(Counter(row.tolist()).most_common(1)[0][0]) for row in x]
        order = np.argsort(key, kind='stable')[::-1]
        return x[order]

    def sort_cols_asc(x: np.ndarray) -> np.ndarray:
        key = [int(Counter(x[:, c].tolist()).most_common(1)[0][0]) for c in range(x.shape[1])]
        order = np.argsort(key, kind='stable')
        return x[:, order]

    def sort_cols_desc(x: np.ndarray) -> np.ndarray:
        key = [int(Counter(x[:, c].tolist()).most_common(1)[0][0]) for c in range(x.shape[1])]
        order = np.argsort(key, kind='stable')[::-1]
        return x[:, order]

    for fn in [sort_rows_asc, sort_rows_desc, sort_cols_asc, sort_cols_desc]:
        if _all_pairs_match(fn, train):
            return fn
    return None


def detect_frame_fill(train: List[Dict]) -> Optional[Callable]:
    """Detect: fill inside of rectangular border with border color."""
    def fill_inside_border(x: np.ndarray) -> np.ndarray:
        bg = _detect_bg(x)
        result = x.copy()
        # Find border color (not bg, appears on edges)
        edges = np.concatenate([x[0,:], x[-1,:], x[:,0], x[:,-1]])
        non_bg_edge = [v for v in edges if v != bg]
        if not non_bg_edge:
            return x
        border_color = int(Counter(non_bg_edge).most_common(1)[0][0])
        # Fill interior
        result[1:-1, 1:-1] = border_color
        return result

    if _all_pairs_match(fill_inside_border, train):
        return fill_inside_border
    return None


# ============================================================================
# RULE PIPELINE
# ============================================================================

DETECTORS = [
    ('geometric',             detect_geometric),
    ('largest_solid_rect',    detect_largest_solid_rect),
    ('keep_multi_px_objects', detect_keep_multi_pixel_objects),
    ('color_map',             detect_color_map),
    ('scale_up',              detect_scale_up),
    ('scale_down',            detect_scale_down),
    ('tiling',                detect_tiling),
    ('extract_object',        detect_extract_object),
    ('gravity',               detect_gravity),
    ('sort_rows/cols',        detect_sort_rows_by_color),
    ('bg_operations',         detect_bg_operations),
    ('frame_fill',            detect_frame_fill),
]


def learn_rule(train: List[Dict]) -> Tuple[str, Callable]:
    """Try each detector; return first that matches all training pairs."""
    for name, detector in DETECTORS:
        fn = detector(train)
        if fn is not None:
            return name, fn
    # Fallback: identity
    return 'identity', lambda x: x.copy()


# ============================================================================
# SOLVER
# ============================================================================

class V50RuleLearner:

    def solve_task(self, task: Dict, task_id: str) -> Tuple[str, List[List]]:
        train = task['train']
        test_items = task['test']

        rule_name, transform = learn_rule(train)

        predictions = []
        for item in test_items:
            inp = np.array(item['input'])
            try:
                pred = transform(inp)
                predictions.append(_to_list(pred))
            except Exception:
                predictions.append(item['input'])

        return rule_name, predictions

    def solve_dataset(self, dataset_path: str, output_path: str):
        with open(dataset_path) as f:
            data = json.load(f)

        submission = {}
        rule_counts: Counter = Counter()

        print(f"\n{'='*70}")
        print("RE-ARC v50: Per-Task Rule Learner")
        print(f"{'='*70}")
        print(f"Solving {len(data)} tasks...\n")

        for i, (task_id, task) in enumerate(data.items()):
            rule_name, preds = self.solve_task(task, task_id)
            rule_counts[rule_name] += 1

            # Format: list of {attempt_1, attempt_2}
            task_preds = []
            for pred in preds:
                task_preds.append({
                    'attempt_1': pred,
                    'attempt_2': pred,
                })
            submission[task_id] = task_preds

            if (i + 1) % 20 == 0:
                print(f"  [{i+1:3d}/{len(data)}] Rules so far: {dict(rule_counts.most_common())}")

        print(f"\n{'='*70}")
        print("Rule distribution:")
        for rule, count in rule_counts.most_common():
            pct = count / len(data) * 100
            print(f"  {rule:20s}: {count:3d} tasks ({pct:.1f}%)")
        print(f"{'='*70}\n")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(submission, f)

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"✅ v50 written: {output_path}")
        print(f"   Tasks: {len(submission)} | Size: {size_mb:.2f}MB")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*10 + "RE-ARC V50: PER-TASK RULE LEARNER" + " "*25 + "║")
    print("╚" + "="*68 + "╝\n")

    # Use latest test challenge file
    challenge_files = sorted(Path('/Users/evanpieser/Desktop/72%/').glob('re-arc_test_challenges-*.json'))
    if not challenge_files:
        print("❌ No test challenge file found in ~/Desktop/72%/")
        print("   Download from: https://arc.markbarney.net/re-arc")
        return False

    dataset_path = str(challenge_files[-1])
    output_path = '/Users/evanpieser/Desktop/72%/octotetrahedral_rearc_v50_rule_learner.json'

    print(f"📂 Input:  {dataset_path}")
    print(f"📤 Output: {output_path}\n")

    solver = V50RuleLearner()
    solver.solve_dataset(dataset_path, output_path)

    # Also copy to submissions/ for git
    submissions_dir = Path('/Users/evanpieser/arc_agi2_submission/submissions')
    submissions_dir.mkdir(exist_ok=True)
    import shutil
    dest = submissions_dir / 'octotetrahedral_rearc_v50_rule_learner.json'
    shutil.copy(output_path, dest)
    print(f"📦 Copied to submissions/: {dest.name}")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
