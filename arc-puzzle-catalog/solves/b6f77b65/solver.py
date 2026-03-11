"""
ARC-AGI puzzle b6f77b65 solver.

The grid contains nested Pi-shaped (Π) rectangles. The indicator color(s) at row 0
select which parts to remove. When an arm or top-bar is removed, the structure
collapses downward according to a hierarchical shift model:
- Removing an arm: the top bar drops by the arm's body height.
- Removing a top bar: creates a 1-row gap; arms above extend through it.
- Shifts cascade through the rectangle hierarchy from bottom to top.
"""

import json
from typing import List, Optional, Tuple, Dict, Any


def detect_pi_shapes(grid: List[List[int]]) -> List[Dict[str, Any]]:
    """Detect all Pi-shaped (upside-down U) rectangles in the grid."""
    rows, cols = len(grid), len(grid[0])
    shapes: List[Dict[str, Any]] = []
    used_tops: set = set()
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r][c] != 0 and (r, c) != (0, 0):
                start = c
                while c < cols and grid[r][c] != 0:
                    c += 1
                end = c - 1
                if end - start >= 2:
                    arm_h = 0
                    for r2 in range(r + 1, rows):
                        left_ok = grid[r2][start] != 0
                        right_ok = grid[r2][end] != 0
                        interior_ok = all(
                            grid[r2][c2] == 0 for c2 in range(start + 1, end)
                        )
                        if left_ok and right_ok and interior_ok:
                            arm_h += 1
                        else:
                            break
                    if arm_h >= 1:
                        key = (r, start, end)
                        if key not in used_tops:
                            used_tops.add(key)
                            shapes.append({
                                'top': r, 'left': start, 'right': end, 'body': arm_h,
                            })
            else:
                c += 1
    return shapes


def find_removed_part(
    grid: List[List[int]], rect: Dict[str, Any], indicator: int
) -> Optional[str]:
    """Determine which part of the rect has the indicator color."""
    r, cl, cr, h = rect['top'], rect['left'], rect['right'], rect['body']
    if all(grid[r + i][cl] == indicator for i in range(h + 1)):
        return 'left'
    if all(grid[r + i][cr] == indicator for i in range(h + 1)):
        return 'right'
    if cr - cl > 1:
        if all(grid[r][c] == indicator for c in range(cl + 1, cr)):
            return 'top'
    return None


def get_shift_at_col(rect: Dict[str, Any], col: int) -> int:
    """Get shift this rect provides at a specific column for rects above."""
    removed = rect['removed']
    if removed == 'left' and col == rect['left']:
        return rect['body'] + 1
    if removed == 'right' and col == rect['right']:
        return rect['body'] + 1
    return rect['bar_shift']


def find_rect_below_at_col(
    rects: List[Dict[str, Any]], below_row: int, col: int
) -> Optional[Dict[str, Any]]:
    """Find the nearest rect at or below below_row that covers the given column."""
    candidates = [r for r in rects if r['top'] >= below_row and r['left'] <= col <= r['right']]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r['top'])
    return candidates[0]


def compute_arm_shift(
    rect: Dict[str, Any], arm_col: int, rects: List[Dict[str, Any]]
) -> Tuple[int, bool]:
    """Compute shift for a rect's arm, handling bypass for removed top bars."""
    below_row = rect['top'] + rect['body'] + 1
    below_rect = find_rect_below_at_col(rects, below_row, arm_col)
    if below_rect is None:
        return 0, False
    if (below_rect['removed'] == 'top'
            and below_rect['left'] < arm_col < below_rect['right']):
        bypass_below_row = below_rect['top'] + below_rect['body'] + 1
        bypass_rect = find_rect_below_at_col(rects, bypass_below_row, arm_col)
        bypass_shift = get_shift_at_col(bypass_rect, arm_col) if bypass_rect else 0
        return bypass_shift, True
    return get_shift_at_col(below_rect, arm_col), False


def transform(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    indicators = {v for v in grid[0] if v != 0}

    if not any(
        grid[r][c] in indicators and (r, c) != (0, 0)
        for r in range(rows) for c in range(cols)
    ):
        return [row[:] for row in grid]

    rects = detect_pi_shapes(grid)
    for rect in rects:
        removed = None
        for ind in indicators:
            r_part = find_removed_part(grid, rect, ind)
            if r_part is not None:
                if removed is None:
                    removed = r_part
                else:
                    removed = 'full'
        rect['removed'] = removed

    rects.sort(key=lambda r: -r['top'])

    for rect in rects:
        ls, ls_bypass = compute_arm_shift(rect, rect['left'], rects)
        rs, rs_bypass = compute_arm_shift(rect, rect['right'], rects)
        rect['left_shift'] = ls
        rect['right_shift'] = rs
        rect['left_bypass'] = ls_bypass
        rect['right_bypass'] = rs_bypass

        h = rect['body']
        removed = rect['removed']

        if ls_bypass:
            bl = find_rect_below_at_col(
                rects, rect['top'] + rect['body'] + 1, rect['left']
            )
            ls_eff = bl['bar_shift'] if bl else 0
        else:
            ls_eff = ls

        if rs_bypass:
            br = find_rect_below_at_col(
                rects, rect['top'] + rect['body'] + 1, rect['right']
            )
            rs_eff = br['bar_shift'] if br else 0
        else:
            rs_eff = rs

        if removed == 'left':
            rect['bar_shift'] = rs_eff + h
        elif removed == 'right':
            rect['bar_shift'] = ls_eff + h
        elif removed == 'top' or removed == 'full':
            rect['bar_shift'] = max(ls_eff, rs_eff) + 1
        else:
            rect['bar_shift'] = max(ls_eff, rs_eff)

        r_top = rect['top']
        cl_r, cr_r = rect['left'], rect['right']
        bar_cells_are_indicator = all(
            grid[r_top][c] in indicators for c in range(cl_r + 1, cr_r)
        ) if cr_r - cl_r > 1 else False
        if bar_cells_are_indicator and removed in ('left', 'right'):
            rect['bar_shift'] += 1

    result = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        result[0][c] = grid[0][c]

    placed = set()

    for rect in rects:
        r, cl, cr, h = rect['top'], rect['left'], rect['right'], rect['body']
        removed = rect['removed']
        ls = rect['left_shift']
        rs = rect['right_shift']
        bs = rect['bar_shift']

        if removed != 'top' and removed != 'full':
            for c in range(cl + 1, cr):
                if grid[r][c] in indicators:
                    placed.add((r, c))
                    continue
                nr = r + bs
                if 0 <= nr < rows:
                    result[nr][c] = grid[r][c]
                placed.add((r, c))

        if removed != 'left' and removed != 'full':
            for i in range(h + 1):
                if grid[r + i][cl] in indicators:
                    placed.add((r + i, cl))
                    continue
                nr = r + i + ls
                if 0 <= nr < rows:
                    result[nr][cl] = grid[r + i][cl]
                placed.add((r + i, cl))
            if rect['left_bypass']:
                nr = r + h + 1 + ls
                if 0 <= nr < rows:
                    result[nr][cl] = grid[r + h][cl]

        if removed != 'right' and removed != 'full':
            for i in range(h + 1):
                if grid[r + i][cr] in indicators:
                    placed.add((r + i, cr))
                    continue
                nr = r + i + rs
                if 0 <= nr < rows:
                    result[nr][cr] = grid[r + i][cr]
                placed.add((r + i, cr))
            if rect['right_bypass']:
                nr = r + h + 1 + rs
                if 0 <= nr < rows:
                    result[nr][cr] = grid[r + h][cr]

        # Mark removed cells as placed
        if removed == 'left':
            for i in range(h + 1):
                placed.add((r + i, cl))
        elif removed == 'right':
            for i in range(h + 1):
                placed.add((r + i, cr))
        elif removed == 'top' or removed == 'full':
            for c in range(cl + 1, cr):
                placed.add((r, c))
            if removed == 'full':
                for i in range(h + 1):
                    placed.add((r + i, cl))
                    placed.add((r + i, cr))

    # Handle orphan cells not placed by any Π — inherit shift from nearest arm above
    for r in range(1, rows):
        for c in range(cols):
            if (r, c) in placed:
                continue
            val = grid[r][c]
            if val == 0 or val in indicators:
                continue
            best_shift = None
            for rect in rects:
                arm_end = rect['top'] + rect['body']
                if c == rect['left'] and arm_end < r and rect['removed'] != 'left':
                    shift = rect['left_shift']
                    if best_shift is None or rect['top'] > best_shift[1]:
                        best_shift = (shift, rect['top'])
                elif c == rect['right'] and arm_end < r and rect['removed'] != 'right':
                    shift = rect['right_shift']
                    if best_shift is None or rect['top'] > best_shift[1]:
                        best_shift = (shift, rect['top'])
            if best_shift is not None:
                nr = r + best_shift[0]
                if 0 <= nr < rows:
                    result[nr][c] = val

    return result


if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/b6f77b65.json") as f:
        task = json.load(f)

    print("=== Training Examples ===")
    all_pass = True
    for ti, ex in enumerate(task['train']):
        predicted = transform(ex['input'])
        match = predicted == ex['output']
        all_pass = all_pass and match
        print(f"Train {ti}: {'PASS' if match else 'FAIL'}")
        if not match:
            for r in range(len(ex['output'])):
                for c in range(len(ex['output'][0])):
                    if predicted[r][c] != ex['output'][r][c]:
                        print(f"  ({r},{c}): pred={predicted[r][c]} exp={ex['output'][r][c]}")

    print(f"\nAll training: {'PASS' if all_pass else 'FAIL'}")

    print("\n=== Test Predictions ===")
    for ti, ex in enumerate(task['test']):
        predicted = transform(ex['input'])
        print(f"\nTest {ti} prediction:")
        for row in predicted:
            print(row)
