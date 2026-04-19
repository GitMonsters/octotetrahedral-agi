from collections import Counter
from typing import List


def transform(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    def get_edge_cells(is_row: bool, idx: int):
        if is_row:
            return [(idx, c, grid[idx][c]) for c in range(cols) if grid[idx][c] != bg]
        else:
            return [(r, idx, grid[r][idx]) for r in range(rows) if grid[r][idx] != bg]

    all_edges = {
        ("row", 0):        get_edge_cells(True,  0),
        ("row", rows - 1): get_edge_cells(True,  rows - 1),
        ("col", 0):        get_edge_cells(False, 0),
        ("col", cols - 1): get_edge_cells(False, cols - 1),
    }

    # Pattern strip: edge with non-9, non-background values.
    pattern_key = None
    for key, cells in all_edges.items():
        if any(v != 9 for _, _, v in cells):
            pattern_key = key
            break

    shift_key = None

    if pattern_key is None:
        # All non-background values are 9s.
        nine_edges = [(k, c) for k, c in all_edges.items() if c]

        if len(nine_edges) == 1:
            # Single edge: pattern strip with no shift strip.
            pattern_key = nine_edges[0][0]
        else:
            # Two adjacent edges: shift strip has M nines (M+1 zones),
            # pattern strip has M+1 nines.
            (k1, c1), (k2, c2) = nine_edges[0], nine_edges[1]
            n1, n2 = len(c1), len(c2)
            if n1 + 1 == n2:
                shift_key, pattern_key = k1, k2
            elif n2 + 1 == n1:
                shift_key, pattern_key = k2, k1
            else:
                # Fallback: row edge = shift strip, col edge = pattern strip.
                shift_key, pattern_key = (k1, k2) if k1[0] == "row" else (k2, k1)
    else:
        # Shift strip: perpendicular adjacent edge with 9s.
        pat_type = pattern_key[0]
        candidates = (
            [("col", 0), ("col", cols - 1)] if pat_type == "row"
            else [("row", 0), ("row", rows - 1)]
        )
        for k in candidates:
            if all_edges.get(k) and any(v == 9 for _, _, v in all_edges[k]):
                shift_key = k
                break

    pat_type, pat_idx = pattern_key
    pat_cells = all_edges[pattern_key]
    # Fill value is the non-9, non-bg value; fallback to 9 if all marks are 9s.
    fill_val = next((v for _, _, v in pat_cells if v != 9), 9)

    # Mark positions along the pattern strip.
    pattern_marks = (
        [c for _, c, _ in pat_cells] if pat_type == "row"
        else [r for r, _, _ in pat_cells]
    )

    if shift_key:
        shift_type, shift_idx_val = shift_key
        shift_cells = all_edges[shift_key]
        shift_9s = (
            [c for _, c, v in shift_cells if v == 9] if shift_type == "row"
            else [r for r, _, v in shift_cells if v == 9]
        )
    else:
        shift_type, shift_idx_val = None, None
        shift_9s = []

    # Propagation direction: away from the pattern strip edge.
    # Shift direction: away from the shift strip edge.
    if pat_type == "row":
        prop_dir = "down" if pat_idx == 0 else "up"
        shift_dir = ("right" if shift_idx_val == 0 else "left") if shift_key else None
    else:
        prop_dir = "right" if pat_idx == 0 else "left"
        shift_dir = ("down" if shift_idx_val == 0 else "up") if shift_key else None

    def get_shift(pos: int) -> int:
        """Cumulative count of 9s in shift strip from pattern strip to pos (inclusive)."""
        if not shift_9s:
            return 0
        if prop_dir in ("down", "right"):
            return sum(1 for s in shift_9s if 1 <= s <= pos)
        else:
            return sum(1 for s in shift_9s if pos <= s <= pat_idx - 1)

    def apply_pattern(out: List[List[int]], pos: int, shift: int) -> None:
        if pat_type == "row":
            for mc in pattern_marks:
                nc = (mc + shift) if shift_dir == "right" else (mc - shift)
                if 0 <= nc < cols:
                    out[pos][nc] = fill_val
        else:
            for mr in pattern_marks:
                nr = (mr + shift) if shift_dir == "down" else (mr - shift)
                if 0 <= nr < rows:
                    out[nr][pos] = fill_val

    out = [[bg] * cols for _ in range(rows)]

    # Preserve pattern strip unchanged.
    if pat_type == "row":
        out[pat_idx] = list(grid[pat_idx])
    else:
        for r in range(rows):
            out[r][pat_idx] = grid[r][pat_idx]

    # Propagate the pattern across the grid.
    if prop_dir == "down":
        for r in range(1, rows):
            apply_pattern(out, r, get_shift(r))
    elif prop_dir == "up":
        for r in range(pat_idx - 1, -1, -1):
            apply_pattern(out, r, get_shift(r))
    elif prop_dir == "left":
        for c in range(pat_idx - 1, -1, -1):
            apply_pattern(out, c, get_shift(c))
    elif prop_dir == "right":
        for c in range(1, cols):
            apply_pattern(out, c, get_shift(c))

    # Preserve shift-strip 9s (written last so they are never overwritten).
    if shift_key:
        if shift_type == "row":
            for c in shift_9s:
                out[shift_idx_val][c] = 9
        else:
            for r in shift_9s:
                out[r][shift_idx_val] = 9

    return out
