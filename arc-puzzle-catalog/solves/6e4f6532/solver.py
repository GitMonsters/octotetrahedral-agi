"""
Solver for ARC-AGI task 6e4f6532.

Pattern: The grid has border strips (2-wide rows/columns of uniform non-bg color)
dividing the space into panels. Inside, there are "objects" — connected blobs of 8s
with internal 9-detail and border-colored decorator pixels attached. There are also
isolated groups of 9s that serve as target positions.

Each object is matched to an isolated 9-group (by count of 9s). The object is
rotated/flipped so its decorator pixels face the correct grid borders, then
translated so its 9s align with the target 9 positions. The decorator colors
are preserved through the transformation.
"""

import json
from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    # Iteratively find border strip cells (uniform rows/cols of non-bg color)
    border_cells = _find_border_cells(grid, bg)

    # Interior non-bg, non-border pixels
    interior_pixels: dict[tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r, c) not in border_cells:
                interior_pixels[(r, c)] = grid[r][c]

    components = _find_cc(interior_pixels)

    main_objects: list[dict[tuple[int, int], int]] = []
    isolated_9_groups: list[set[tuple[int, int]]] = []
    for comp in components:
        comp_dict = {pos: interior_pixels[pos] for pos in comp}
        vals = set(comp_dict.values())
        if 8 in vals:
            main_objects.append(comp_dict)
        elif vals == {9}:
            isolated_9_groups.append(comp)

    # Border reference: border cells + structural interior, grouped by color
    obj_positions = set()
    for obj in main_objects:
        obj_positions.update(obj.keys())
    iso_positions = set()
    for group in isolated_9_groups:
        iso_positions.update(group)

    border_ref: dict[int, list[tuple[int, int]]] = {}
    # Include border strip cells
    for (r, c) in border_cells:
        v = grid[r][c]
        border_ref.setdefault(v, []).append((r, c))
    # Include structural interior components (non-object, non-isolated)
    for (r, c), v in interior_pixels.items():
        if (r, c) not in obj_positions and (r, c) not in iso_positions:
            border_ref.setdefault(v, []).append((r, c))

    matches = _match_objects_to_targets(main_objects, isolated_9_groups)

    output = [row[:] for row in grid]
    for obj in main_objects:
        for (r, c) in obj:
            output[r][c] = bg
    for group in isolated_9_groups:
        for (r, c) in group:
            output[r][c] = bg

    for obj, target_group in matches:
        result = _transform_object(obj, target_group, border_ref)
        if result:
            for (r, c), v in result.items():
                if 0 <= r < rows and 0 <= c < cols:
                    output[r][c] = v

    return output


def _find_border_cells(grid, bg) -> set[tuple[int, int]]:
    """Iteratively find border strip cells (uniform rows/cols of non-bg color)."""
    rows = len(grid)
    cols = len(grid[0])
    border = set()

    changed = True
    while changed:
        changed = False
        # Check rows: all non-border cells in this row have the same non-bg color
        for r in range(rows):
            active_cols = [c for c in range(cols) if (r, c) not in border]
            if not active_cols:
                continue
            vals = set(grid[r][c] for c in active_cols)
            if len(vals) == 1 and vals.pop() != bg:
                for c in active_cols:
                    if (r, c) not in border:
                        border.add((r, c))
                        changed = True
        # Check cols
        for c in range(cols):
            active_rows = [r for r in range(rows) if (r, c) not in border]
            if not active_rows:
                continue
            vals = set(grid[r][c] for r in active_rows)
            if len(vals) == 1 and vals.pop() != bg:
                for r in active_rows:
                    if (r, c) not in border:
                        border.add((r, c))
                        changed = True
    return border


def _find_cc(pixels: dict) -> list[set]:
    remaining = set(pixels.keys())
    components = []
    while remaining:
        start = next(iter(remaining))
        component: set[tuple[int, int]] = set()
        queue = [start]
        while queue:
            pos = queue.pop()
            if pos in component:
                continue
            component.add(pos)
            remaining.discard(pos)
            r, c = pos
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    npos = (r + dr, c + dc)
                    if npos in remaining:
                        queue.append(npos)
        components.append(component)
    return components


def _match_objects_to_targets(
    main_objects: list[dict], isolated_9_groups: list[set]
) -> list[tuple[dict, set]]:
    matches = []
    used = set()

    obj_nine_counts = []
    for obj in main_objects:
        n9 = sum(1 for v in obj.values() if v == 9)
        obj_nine_counts.append(n9)

    group_sizes = [len(g) for g in isolated_9_groups]

    # Try unique matching first
    for oi, obj in enumerate(main_objects):
        n9 = obj_nine_counts[oi]
        for gi, group in enumerate(isolated_9_groups):
            if gi not in used and group_sizes[gi] == n9:
                matches.append((obj, group))
                used.add(gi)
                break

    return matches


# 8 rigid transforms: (dr, dc) -> (new_dr, new_dc)
_TRANSFORMS = [
    lambda dr, dc: (dr, dc),       # 0: identity
    lambda dr, dc: (dc, -dr),      # 1: rot90cw
    lambda dr, dc: (-dr, -dc),     # 2: rot180
    lambda dr, dc: (-dc, dr),      # 3: rot270cw
    lambda dr, dc: (dr, -dc),      # 4: flipH
    lambda dr, dc: (-dr, dc),      # 5: flipV
    lambda dr, dc: (dc, dr),       # 6: transpose
    lambda dr, dc: (-dc, -dr),     # 7: anti-transpose
]

# How each transform maps cardinal sides
_SIDE_MAP = [
    {"T": "T", "B": "B", "L": "L", "R": "R"},  # identity
    {"T": "R", "B": "L", "L": "T", "R": "B"},  # rot90cw
    {"T": "B", "B": "T", "L": "R", "R": "L"},  # rot180
    {"T": "L", "B": "R", "L": "B", "R": "T"},  # rot270cw
    {"T": "T", "B": "B", "L": "R", "R": "L"},  # flipH
    {"T": "B", "B": "T", "L": "L", "R": "R"},  # flipV
    {"T": "L", "B": "R", "L": "T", "R": "B"},  # transpose
    {"T": "R", "B": "L", "L": "B", "R": "T"},  # anti-transpose
]

_IS_ROTATION = [True, True, True, True, False, False, False, False]


def _get_side(dr: float, dc: float) -> str:
    if abs(dr) >= abs(dc):
        return "T" if dr < 0 else "B"
    else:
        return "L" if dc < 0 else "R"


def _get_border_side(
    border_positions: list[tuple[int, int]], target_cr: float, target_cc: float
) -> str:
    """Determine if border is row-like or col-like, then get target side."""
    min_r = min(r for r, c in border_positions)
    max_r = max(r for r, c in border_positions)
    min_c = min(c for r, c in border_positions)
    max_c = max(c for r, c in border_positions)

    height = max_r - min_r + 1
    width = max_c - min_c + 1

    if width >= height:  # Horizontal (row) border
        border_mid_r = (min_r + max_r) / 2
        return "T" if border_mid_r < target_cr else "B"
    else:  # Vertical (column) border
        border_mid_c = (min_c + max_c) / 2
        return "L" if border_mid_c < target_cc else "R"


def _centroid(positions):
    n = len(positions)
    return (
        sum(r for r, c in positions) / n,
        sum(c for r, c in positions) / n,
    )


def _transform_object(
    obj_pixels: dict[tuple[int, int], int],
    target_9_positions: set[tuple[int, int]],
    border_ref: dict[int, list[tuple[int, int]]],
) -> dict[tuple[int, int], int] | None:
    obj_9s = [(r, c) for (r, c), v in obj_pixels.items() if v == 9]

    target_9s = list(target_9_positions)
    target_cr, target_cc = _centroid(target_9s)

    # Current decorator sides: use 4-neighbor direction from blob to decorator
    blob_set = frozenset((r, c) for (r, c), v in obj_pixels.items() if v in (8, 9))
    decorator_groups: dict[int, list[tuple[int, int]]] = {}
    for (r, c), v in obj_pixels.items():
        if v not in (8, 9):
            decorator_groups.setdefault(v, []).append((r, c))

    current_sides: dict[int, str] = {}
    for color, positions in decorator_groups.items():
        dr_sum, dc_sum = 0, 0
        for r, c in positions:
            for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if (nr, nc) in blob_set:
                    dr_sum += r - nr
                    dc_sum += c - nc
        if dr_sum != 0 or dc_sum != 0:
            current_sides[color] = _get_side(dr_sum, dc_sum)

    # Target sides: determine if each border is row-like or col-like,
    # then use the perpendicular axis to determine direction
    target_sides: dict[int, str] = {}
    for color in current_sides:
        if color in border_ref and border_ref[color]:
            target_sides[color] = _get_border_side(
                border_ref[color], target_cr, target_cc
            )

    # Find valid transforms
    candidates: list[tuple[int, dict[tuple[int, int], int]]] = []

    for tid in range(8):
        # Check side mapping
        side_ok = True
        for color in current_sides:
            if color in target_sides:
                mapped = _SIDE_MAP[tid].get(current_sides[color])
                if mapped != target_sides[color]:
                    side_ok = False
                    break
        if not side_ok:
            continue

        # Check 9 alignment
        center = obj_9s[0]
        transformed_rel: dict[tuple[int, int], int] = {}
        for (r, c), v in obj_pixels.items():
            dr, dc = r - center[0], c - center[1]
            new_dr, new_dc = _TRANSFORMS[tid](dr, dc)
            transformed_rel[(new_dr, new_dc)] = v

        trans_9s_rel = [(dr, dc) for (dr, dc), v in transformed_rel.items() if v == 9]

        found = False
        best_shifted = None

        for t9_rel in trans_9s_rel:
            for tgt in target_9s:
                sr = tgt[0] - t9_rel[0]
                sc = tgt[1] - t9_rel[1]

                shifted: dict[tuple[int, int], int] = {}
                for (dr, dc), v in transformed_rel.items():
                    shifted[(dr + sr, dc + sc)] = v

                shifted_9s = frozenset(pos for pos, v in shifted.items() if v == 9)
                if shifted_9s == frozenset(target_9s):
                    best_shifted = shifted
                    found = True
                    break
            if found:
                break

        if found and best_shifted is not None:
            candidates.append((tid, best_shifted))

    if not candidates:
        return None

    # Prefer pure rotations over reflections as tiebreaker
    for tid, shifted in candidates:
        if _IS_ROTATION[tid]:
            return shifted

    return candidates[0][1]


if __name__ == "__main__":
    import os

    task_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "dataset",
        "tasks",
        "6e4f6532.json",
    )
    with open(task_path) as f:
        data = json.load(f)

    all_pass = True
    for i, pair in enumerate(data["train"]):
        result = solve(pair["input"])
        expected = pair["output"]
        ok = result == expected
        print(f"Train pair {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if r < len(result) and c < len(result[0]):
                        if result[r][c] != expected[r][c]:
                            print(
                                f"  Mismatch at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}"
                            )

    for i, pair in enumerate(data.get("test", [])):
        result = solve(pair["input"])
        if "output" in pair:
            expected = pair["output"]
            ok = result == expected
            print(f"Test pair {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        if r < len(result) and c < len(result[0]):
                            if result[r][c] != expected[r][c]:
                                print(
                                    f"  Mismatch at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}"
                                )
        else:
            print(f"Test pair {i}: output produced (no expected to compare)")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
