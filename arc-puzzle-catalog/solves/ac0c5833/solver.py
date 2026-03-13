"""
ARC-AGI task ac0c5833 solver.

Pattern: Each grid has groups of three 4s forming L-shapes (3 corners of a
2-gap rectangle). One L-shape has a colored 2-shape at its "open" corner
(the template). The task copies/transforms that 2-shape to every other
L-shape's open corner, reflecting it to match the L's orientation.
"""
import json
import copy
from typing import List, Tuple, Set, Optional

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    output = copy.deepcopy(grid)

    fours: Set[Tuple[int, int]] = set()
    twos: Set[Tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                fours.add((r, c))
            elif grid[r][c] == 2:
                twos.add((r, c))

    # Find L-shapes: 3 corners of a rectangle with row-gap=2, col-gap=2
    l_shapes = []
    for r in range(rows - 2):
        for c in range(cols - 2):
            corners = [(r, c), (r, c + 2), (r + 2, c), (r + 2, c + 2)]
            present = tuple(p for p in corners if p in fours)
            absent = [p for p in corners if p not in fours]
            if len(present) == 3:
                l_shapes.append((present, absent[0]))

    # Deduplicate by frozenset of present fours
    seen = set()
    unique = []
    for present, missing in l_shapes:
        key = (frozenset(present), missing)
        if key not in seen:
            seen.add(key)
            unique.append((present, missing))
    l_shapes = unique

    # Template L: the one closest to the 2-shape
    template = None
    best_dist = float('inf')
    for ls in l_shapes:
        _, missing = ls
        mr, mc = missing
        for tr, tc in twos:
            d = abs(tr - mr) + abs(tc - mc)
            if d < best_dist:
                best_dist = d
                template = ls

    if template is None:
        return output

    # Template direction (which corner of rectangle is missing)
    mr, mc = template[1]
    all_pts = list(template[0]) + [template[1]]
    min_r = min(p[0] for p in all_pts)
    max_r = max(p[0] for p in all_pts)
    min_c = min(p[1] for p in all_pts)
    max_c = max(p[1] for p in all_pts)
    row_dir = -1 if mr == min_r else 1
    col_dir = -1 if mc == min_c else 1

    # Extract canonical 2-shape (normalized so offsets are non-negative)
    canonical = []
    for r, c in twos:
        dr = (r - mr) * row_dir
        dc = (c - mc) * col_dir
        if dr >= 0 and dc >= 0:
            canonical.append((dr, dc))

    # Place shape at each non-template L-shape
    for ls in l_shapes:
        if ls == template:
            continue
        _, missing_t = ls
        mr_t, mc_t = missing_t
        all_pts_t = list(ls[0]) + [missing_t]
        min_r_t = min(p[0] for p in all_pts_t)
        min_c_t = min(p[1] for p in all_pts_t)
        row_dir_t = -1 if mr_t == min_r_t else 1
        col_dir_t = -1 if mc_t == min_c_t else 1

        for can_dr, can_dc in canonical:
            new_r = mr_t + can_dr * row_dir_t
            new_c = mc_t + can_dc * col_dir_t
            if 0 <= new_r < rows and 0 <= new_c < cols:
                output[new_r][new_c] = 2

    return output


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/ac0c5833.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        expected = pair["output"]
        match = result == expected
        all_pass = all_pass and match
        if not match:
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Train {i} mismatch at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")

    # Solve test
    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        print(f"\nTest {i} output:")
        for row in result:
            print(row)

    print(f"\nAll train passed: {all_pass}")
