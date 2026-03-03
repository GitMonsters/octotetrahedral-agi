"""
ARC Post-Processor: Fix ConvNet near-misses using output constraints.

Strategy:
1. Learn constraints from training I/O pairs (color counts, symmetry, patterns)
2. Get ConvNet prediction (typically 94-98.5% correct)
3. Identify cells that violate constraints
4. Search for corrections that satisfy all constraints
"""
import numpy as np
import json
import os
from itertools import product, combinations
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Any


def get_color_counts(grid: np.ndarray) -> Dict[int, int]:
    """Count occurrences of each color."""
    return dict(Counter(grid.flatten()))


def get_row_color_counts(grid: np.ndarray) -> List[Dict[int, int]]:
    """Color counts per row."""
    return [dict(Counter(row)) for row in grid]


def get_col_color_counts(grid: np.ndarray) -> List[Dict[int, int]]:
    """Color counts per column."""
    return [dict(Counter(grid[:, j])) for j in range(grid.shape[1])]


def check_row_col_consistency(grids: List[np.ndarray]) -> Dict[str, Any]:
    """Check if all output grids have same row/col color distributions."""
    if len(grids) < 2:
        return {}
    
    constraints = {}
    
    # Check if same total color counts across outputs
    counts = [get_color_counts(g) for g in grids]
    # Normalize by grid size
    h0, w0 = grids[0].shape
    all_same_size = all(g.shape == (h0, w0) for g in grids)
    
    if all_same_size:
        # Check if per-row color counts follow a pattern
        row_counts = [get_row_color_counts(g) for g in grids]
        # Check if color count per row is constant
        row_totals = []
        for g_rows in row_counts:
            totals = [frozenset(rc.items()) for rc in g_rows]
            row_totals.append(totals)
    
    return constraints


def has_horizontal_symmetry(grid: np.ndarray) -> bool:
    """Check if grid is horizontally symmetric (left-right mirror)."""
    return np.array_equal(grid, grid[:, ::-1])


def has_vertical_symmetry(grid: np.ndarray) -> bool:
    """Check if grid is vertically symmetric (top-bottom mirror)."""
    return np.array_equal(grid, grid[::-1, :])


def has_rotational_symmetry(grid: np.ndarray) -> bool:
    """Check if grid has 180-degree rotational symmetry."""
    return np.array_equal(grid, grid[::-1, ::-1])


def has_diagonal_symmetry(grid: np.ndarray) -> bool:
    """Check if grid equals its transpose."""
    h, w = grid.shape
    if h != w:
        return False
    return np.array_equal(grid, grid.T)


def get_period(row: np.ndarray) -> Optional[int]:
    """Find the repeating period of a 1D array."""
    n = len(row)
    for p in range(1, n // 2 + 1):
        if n % p == 0:
            if all(row[i] == row[i % p] for i in range(n)):
                return p
    return None


def learn_output_constraints(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """Learn constraints that ALL training outputs satisfy."""
    if not pairs:
        return {}
    
    outputs = [p[1] for p in pairs]
    inputs = [p[0] for p in pairs]
    constraints = {}
    
    # Only analyze same-size I/O
    same_size = all(inp.shape == out.shape for inp, out in pairs)
    if not same_size:
        return constraints
    
    # 1. Symmetry constraints
    h_sym = all(has_horizontal_symmetry(o) for o in outputs)
    v_sym = all(has_vertical_symmetry(o) for o in outputs)
    r_sym = all(has_rotational_symmetry(o) for o in outputs)
    d_sym = all(has_diagonal_symmetry(o) for o in outputs)
    
    if h_sym:
        constraints['h_symmetry'] = True
    if v_sym:
        constraints['v_symmetry'] = True
    if r_sym:
        constraints['r_symmetry'] = True
    if d_sym:
        constraints['d_symmetry'] = True
    
    # 2. Color palette constraints
    all_colors = set()
    for o in outputs:
        all_colors.update(o.flatten())
    constraints['palette'] = all_colors
    
    # 3. Per-color count relationships with input
    # Check if output has same count of each color as input
    same_color_counts = True
    for inp, out in pairs:
        if get_color_counts(inp) != get_color_counts(out):
            same_color_counts = False
            break
    if same_color_counts:
        constraints['same_color_counts_as_input'] = True
    
    # 4. Check if unchanged cells remain unchanged
    # (cells where input == output are never changed)
    unchanged_masks = []
    for inp, out in pairs:
        unchanged_masks.append(inp == out)
    constraints['_pairs'] = pairs  # Store for later use
    
    # 5. Row/column color count constraints
    # Check if every row has same set of color counts
    row_count_patterns = []
    for o in outputs:
        row_patterns = []
        for row in o:
            counts = tuple(sorted(Counter(row).values()))
            row_patterns.append(counts)
        row_count_patterns.append(row_patterns)
    
    # Check if all outputs have same row pattern structure
    if len(set(len(p) for p in row_count_patterns)) == 1:
        # All same number of rows
        n_rows = len(row_count_patterns[0])
        consistent_rows = True
        for r in range(n_rows):
            patterns = [rcp[r] for rcp in row_count_patterns]
            if len(set(patterns)) > 1:
                consistent_rows = False
                break
        if consistent_rows and n_rows > 0:
            constraints['row_count_pattern'] = row_count_patterns[0]
    
    # Same for columns
    col_count_patterns = []
    for o in outputs:
        col_patterns = []
        for j in range(o.shape[1]):
            counts = tuple(sorted(Counter(o[:, j]).values()))
            col_patterns.append(counts)
        col_count_patterns.append(col_patterns)
    
    if len(set(len(p) for p in col_count_patterns)) == 1:
        n_cols = len(col_count_patterns[0])
        consistent_cols = True
        for c in range(n_cols):
            patterns = [ccp[c] for ccp in col_count_patterns]
            if len(set(patterns)) > 1:
                consistent_cols = False
                break
        if consistent_cols and n_cols > 0:
            constraints['col_count_pattern'] = col_count_patterns[0]
    
    # 6. Periodic row/column patterns
    row_periods = []
    for o in outputs:
        periods = []
        for row in o:
            p = get_period(row)
            periods.append(p)
        row_periods.append(periods)
    
    # 7. Connected component count constraints
    # Check if all outputs have same number of connected components per color
    
    return constraints


def find_violated_cells(
    prediction: np.ndarray,
    constraints: Dict[str, Any],
    test_input: np.ndarray
) -> List[Tuple[int, int, List[int]]]:
    """Find cells in prediction that violate constraints.
    Returns list of (row, col, candidate_colors) for violated cells.
    """
    violations = []
    h, w = prediction.shape
    palette = constraints.get('palette', set(range(10)))
    
    # Check symmetry violations
    if constraints.get('h_symmetry'):
        for i in range(h):
            for j in range(w // 2):
                mirror_j = w - 1 - j
                if prediction[i, j] != prediction[i, mirror_j]:
                    # One of them is wrong
                    violations.append((i, j, [prediction[i, mirror_j]]))
                    violations.append((i, mirror_j, [prediction[i, j]]))
    
    if constraints.get('v_symmetry'):
        for i in range(h // 2):
            mirror_i = h - 1 - i
            for j in range(w):
                if prediction[i, j] != prediction[mirror_i, j]:
                    violations.append((i, j, [prediction[mirror_i, j]]))
                    violations.append((mirror_i, j, [prediction[i, j]]))
    
    if constraints.get('r_symmetry'):
        for i in range(h):
            for j in range(w):
                ri, rj = h - 1 - i, w - 1 - j
                if (ri, rj) > (i, j) and prediction[i, j] != prediction[ri, rj]:
                    violations.append((i, j, [prediction[ri, rj]]))
                    violations.append((ri, rj, [prediction[i, j]]))
    
    if constraints.get('d_symmetry'):
        for i in range(h):
            for j in range(min(i, w)):
                if prediction[i, j] != prediction[j, i]:
                    violations.append((i, j, [prediction[j, i]]))
                    violations.append((j, i, [prediction[i, j]]))
    
    # Check palette violations
    for i in range(h):
        for j in range(w):
            if prediction[i, j] not in palette:
                violations.append((i, j, list(palette)))
    
    # Check color count violations
    if constraints.get('same_color_counts_as_input') and test_input.shape == prediction.shape:
        expected_counts = get_color_counts(test_input)
        actual_counts = get_color_counts(prediction)
        for color in set(list(expected_counts.keys()) + list(actual_counts.keys())):
            exp = expected_counts.get(color, 0)
            act = actual_counts.get(color, 0)
            if exp != act:
                # Some cells need to change to/from this color
                pass  # Hard to identify which specific cells
    
    # Deduplicate
    seen = set()
    unique_violations = []
    for r, c, colors in violations:
        if (r, c) not in seen:
            seen.add((r, c))
            unique_violations.append((r, c, colors))
    
    return unique_violations


def apply_symmetry_fix(
    prediction: np.ndarray,
    constraints: Dict[str, Any],
    confidence: Optional[np.ndarray] = None
) -> np.ndarray:
    """Fix prediction using symmetry constraints.
    Uses confidence scores to decide which cell to keep when there's a conflict.
    """
    result = prediction.copy()
    h, w = result.shape
    
    if constraints.get('h_symmetry'):
        for i in range(h):
            for j in range(w // 2):
                mirror_j = w - 1 - j
                if result[i, j] != result[i, mirror_j]:
                    if confidence is not None:
                        if confidence[i, j] >= confidence[i, mirror_j]:
                            result[i, mirror_j] = result[i, j]
                        else:
                            result[i, j] = result[i, mirror_j]
                    else:
                        # Default: keep left side
                        result[i, mirror_j] = result[i, j]
    
    if constraints.get('v_symmetry'):
        for i in range(h // 2):
            mirror_i = h - 1 - i
            for j in range(w):
                if result[i, j] != result[mirror_i, j]:
                    if confidence is not None:
                        if confidence[i, j] >= confidence[mirror_i, j]:
                            result[mirror_i, j] = result[i, j]
                        else:
                            result[i, j] = result[mirror_i, j]
                    else:
                        result[mirror_i, j] = result[i, j]
    
    if constraints.get('r_symmetry'):
        for i in range(h):
            for j in range(w):
                ri, rj = h - 1 - i, w - 1 - j
                if (ri, rj) > (i, j) and result[i, j] != result[ri, rj]:
                    if confidence is not None:
                        if confidence[i, j] >= confidence[ri, rj]:
                            result[ri, rj] = result[i, j]
                        else:
                            result[i, j] = result[ri, rj]
                    else:
                        result[ri, rj] = result[i, j]
    
    if constraints.get('d_symmetry'):
        for i in range(h):
            for j in range(min(i, w)):
                if result[i, j] != result[j, i]:
                    if confidence is not None:
                        if confidence[i, j] >= confidence[j, i]:
                            result[j, i] = result[i, j]
                        else:
                            result[i, j] = result[j, i]
                    else:
                        result[j, i] = result[i, j]
    
    return result


def apply_color_count_fix(
    prediction: np.ndarray,
    test_input: np.ndarray,
    confidence: Optional[np.ndarray] = None
) -> np.ndarray:
    """Fix prediction to match input's color counts (if constraint applies)."""
    result = prediction.copy()
    expected = get_color_counts(test_input)
    actual = get_color_counts(result)
    
    # Find colors that have too many/few
    excess = {}  # color -> count to remove
    deficit = {}  # color -> count to add
    
    for color in set(list(expected.keys()) + list(actual.keys())):
        exp = expected.get(color, 0)
        act = actual.get(color, 0)
        if act > exp:
            excess[color] = act - exp
        elif act < exp:
            deficit[color] = exp - act
    
    if not excess or not deficit:
        return result
    
    # Find cells with excess colors, sorted by lowest confidence
    excess_cells = []
    for color, count in excess.items():
        cells = []
        h, w = result.shape
        for i in range(h):
            for j in range(w):
                if result[i, j] == color:
                    conf = confidence[i, j] if confidence is not None else 0.5
                    cells.append((conf, i, j))
        cells.sort()  # Lowest confidence first
        excess_cells.extend([(i, j, color) for _, i, j in cells[:count]])
    
    # Assign deficit colors to excess cells
    deficit_list = []
    for color, count in deficit.items():
        deficit_list.extend([color] * count)
    
    for idx, (i, j, old_color) in enumerate(excess_cells):
        if idx < len(deficit_list):
            result[i, j] = deficit_list[idx]
    
    return result


def learn_local_rules(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict]:
    """Learn local cell-level rules from training pairs.
    
    For each changed cell, look at the local context (3x3 neighborhood in input)
    and learn what color transformation happened.
    """
    if not pairs:
        return None
    
    if not all(inp.shape == out.shape for inp, out in pairs):
        return None
    
    # Collect (neighborhood, output_color) examples for changed cells
    rules = defaultdict(Counter)  # neighborhood_key -> {output_color: count}
    
    for inp, out in pairs:
        h, w = inp.shape
        for i in range(h):
            for j in range(w):
                if inp[i, j] != out[i, j]:
                    # Get 3x3 neighborhood
                    nb = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                nb.append(inp[ni, nj])
                            else:
                                nb.append(-1)
                    key = tuple(nb)
                    rules[key][out[i, j]] += 1
    
    if not rules:
        return None
    
    # Only keep rules that are consistent (same neighborhood always -> same output)
    consistent_rules = {}
    for key, counter in rules.items():
        if len(counter) == 1:
            color, count = list(counter.items())[0]
            consistent_rules[key] = color
    
    return consistent_rules if consistent_rules else None


def apply_local_rules(
    prediction: np.ndarray,
    test_input: np.ndarray,
    rules: Dict
) -> np.ndarray:
    """Apply learned local rules to correct prediction."""
    result = prediction.copy()
    h, w = test_input.shape
    
    for i in range(h):
        for j in range(w):
            nb = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        nb.append(test_input[ni, nj])
                    else:
                        nb.append(-1)
            key = tuple(nb)
            if key in rules:
                result[i, j] = rules[key]
    
    return result


def brute_force_near_miss(
    prediction: np.ndarray,
    confidence: np.ndarray,
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    max_cells: int = 8,
    max_candidates: int = 100000
) -> List[np.ndarray]:
    """Brute-force search for corrections on low-confidence cells.
    
    Strategy:
    1. Find the N lowest-confidence cells
    2. For each, get top-2 predicted colors
    3. Try all 2^N combinations
    4. Validate each against learned constraints
    """
    h, w = prediction.shape
    
    # Get per-cell confidence and alternative predictions
    # confidence shape: (10, H, W) - softmax probabilities
    if confidence.ndim == 3 and confidence.shape[0] == 10:
        probs = confidence
    else:
        return [prediction]
    
    # Find cells where model is least confident
    top_prob = probs.max(axis=0)  # (H, W)
    top_color = probs.argmax(axis=0)  # (H, W)
    
    # Get second-best color for each cell
    sorted_probs = np.sort(probs, axis=0)[::-1]  # Sorted descending
    sorted_colors = np.argsort(probs, axis=0)[::-1]
    second_color = sorted_colors[1]  # (H, W)
    second_prob = sorted_probs[1]  # (H, W)
    
    # Find uncertain cells (lowest ratio of top to second prob)
    ratio = np.where(second_prob > 0, top_prob / (second_prob + 1e-10), 1000.0)
    
    # Flatten and sort
    cells = []
    for i in range(h):
        for j in range(w):
            cells.append((ratio[i, j], i, j, top_color[i, j], second_color[i, j]))
    cells.sort()  # Most uncertain first
    
    uncertain = cells[:max_cells]
    
    if not uncertain:
        return [prediction]
    
    # Generate candidates by flipping uncertain cells
    candidates = []
    n = len(uncertain)
    
    for bits in range(2**n):
        if len(candidates) >= max_candidates:
            break
        
        candidate = prediction.copy()
        for k in range(n):
            if bits & (1 << k):
                _, i, j, _, alt = uncertain[k]
                candidate[i, j] = alt
        candidates.append(candidate)
    
    return candidates


def validate_candidate(
    candidate: np.ndarray,
    constraints: Dict[str, Any],
    test_input: np.ndarray
) -> float:
    """Score a candidate against constraints. Higher = better."""
    score = 0.0
    
    # Symmetry checks
    if constraints.get('h_symmetry'):
        if has_horizontal_symmetry(candidate):
            score += 10.0
    
    if constraints.get('v_symmetry'):
        if has_vertical_symmetry(candidate):
            score += 10.0
    
    if constraints.get('r_symmetry'):
        if has_rotational_symmetry(candidate):
            score += 10.0
    
    if constraints.get('d_symmetry'):
        if has_diagonal_symmetry(candidate):
            score += 10.0
    
    # Color count check
    if constraints.get('same_color_counts_as_input') and test_input.shape == candidate.shape:
        if get_color_counts(test_input) == get_color_counts(candidate):
            score += 20.0
    
    # Palette check
    palette = constraints.get('palette')
    if palette:
        used_colors = set(candidate.flatten())
        if used_colors.issubset(palette):
            score += 5.0
    
    # Row count pattern check
    row_pattern = constraints.get('row_count_pattern')
    if row_pattern and len(row_pattern) == candidate.shape[0]:
        match = True
        for r, expected in enumerate(row_pattern):
            actual = tuple(sorted(Counter(candidate[r]).values()))
            if actual != expected:
                match = False
                break
        if match:
            score += 15.0
    
    # Col count pattern check
    col_pattern = constraints.get('col_count_pattern')
    if col_pattern and len(col_pattern) == candidate.shape[1]:
        match = True
        for c, expected in enumerate(col_pattern):
            actual = tuple(sorted(Counter(candidate[:, c]).values()))
            if actual != expected:
                match = False
                break
        if match:
            score += 15.0
    
    return score


def postprocess_prediction(
    prediction: np.ndarray,
    confidence: Optional[np.ndarray],
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_input: np.ndarray,
    verbose: bool = False
) -> List[np.ndarray]:
    """Main post-processing pipeline.
    
    Returns list of candidate predictions, best first.
    """
    results = [prediction.copy()]
    
    # 1. Learn constraints
    constraints = learn_output_constraints(pairs)
    if verbose and constraints:
        c_names = [k for k in constraints if not k.startswith('_')]
        print(f"    Constraints learned: {c_names}")
    
    # 2. Apply symmetry fixes
    sym_fixed = apply_symmetry_fix(
        prediction, constraints,
        confidence.max(axis=0) if confidence is not None and confidence.ndim == 3 else None
    )
    if not np.array_equal(sym_fixed, prediction):
        results.append(sym_fixed)
        if verbose:
            diff = (sym_fixed != prediction).sum()
            print(f"    Symmetry fix: {diff} cells changed")
    
    # 3. Apply color count fix
    if constraints.get('same_color_counts_as_input'):
        cc_fixed = apply_color_count_fix(
            prediction, test_input,
            confidence.max(axis=0) if confidence is not None and confidence.ndim == 3 else None
        )
        if not np.array_equal(cc_fixed, prediction):
            results.append(cc_fixed)
            if verbose:
                diff = (cc_fixed != prediction).sum()
                print(f"    Color count fix: {diff} cells changed")
    
    # 4. Apply local rules
    local_rules = learn_local_rules(pairs)
    if local_rules:
        lr_fixed = apply_local_rules(prediction, test_input, local_rules)
        if not np.array_equal(lr_fixed, prediction):
            results.append(lr_fixed)
            if verbose:
                diff = (lr_fixed != prediction).sum()
                print(f"    Local rule fix: {diff} cells changed")
    
    # 5. Brute-force search on uncertain cells
    if confidence is not None and confidence.ndim == 3:
        candidates = brute_force_near_miss(
            prediction, confidence, pairs,
            max_cells=min(8, max(6, 8)),
            max_candidates=10000
        )
        
        # Score and rank candidates
        scored = []
        for cand in candidates:
            score = validate_candidate(cand, constraints, test_input)
            scored.append((score, cand))
        
        scored.sort(key=lambda x: -x[0])
        
        # Add top unique candidates
        seen = set()
        for score, cand in scored[:20]:
            key = cand.tobytes()
            if key not in seen:
                seen.add(key)
                if not np.array_equal(cand, prediction):
                    results.append(cand)
    
    # 6. Combine: symmetry fix + brute-force
    if len(results) > 1 and confidence is not None and confidence.ndim == 3:
        for base in results[1:3]:  # Try symmetry-fixed and color-count-fixed
            candidates = brute_force_near_miss(
                base, confidence, pairs,
                max_cells=6,
                max_candidates=5000
            )
            scored = []
            for cand in candidates:
                score = validate_candidate(cand, constraints, test_input)
                scored.append((score, cand))
            scored.sort(key=lambda x: -x[0])
            for score, cand in scored[:5]:
                key = cand.tobytes()
                if key not in {r.tobytes() for r in results}:
                    results.append(cand)
    
    return results


def test_postprocessor():
    """Test on a few eval tasks to see if post-processing helps."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    data_dir = os.path.expanduser("~/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation")
    
    # Test on tasks where ConvNet was closest
    best_tasks = [
        "88e364bc",  # 98.5% (6 wrong on 20x20)
        "409aa875",  # 96.5% (9 wrong on 16x16)
        "dd6b8c4b",  # 93.4% (8 wrong on 11x11)
        "8e5c0c38",  # 97.7% (11 wrong on 22x22)
        "135a2760",  # 97.0% (25 wrong on 22x22)
    ]
    
    for task_id in best_tasks:
        fpath = os.path.join(data_dir, f"{task_id}.json")
        if not os.path.exists(fpath):
            continue
        
        with open(fpath) as f:
            task = json.load(f)
        
        pairs = [(np.array(ex['input']), np.array(ex['output'])) for ex in task['train']]
        
        # Learn constraints
        constraints = learn_output_constraints(pairs)
        c_names = [k for k in constraints if not k.startswith('_')]
        print(f"\n{task_id}: constraints = {c_names}")
        
        for k, v in constraints.items():
            if k.startswith('_'):
                continue
            if k == 'palette':
                print(f"  palette: {sorted(v)}")
            elif k in ('row_count_pattern', 'col_count_pattern'):
                print(f"  {k}: {v[:3]}...")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    test_postprocessor()
