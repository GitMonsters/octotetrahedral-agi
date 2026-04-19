"""
ARC-AGI puzzle c7f57c3e solver.

Pattern: Each grid contains multiple objects sharing the same arm structure (color 1)
at different scales. Objects come in exactly two "variants" distinguished by their
non-arm colored cells (center color, extra element position/color). The transformation
swaps every object's variant to the other one.

Algorithm:
1. Flood-fill (8-connected) to find objects
2. For each object, compute arm bounding box and scale
3. Normalize non-arm cells to a unit template via integer division by scale
4. Group objects into two variant types
5. Redraw each object using the other variant's template (scaled back up)
"""

import json
import copy
from collections import Counter, defaultdict


def transform(grid):
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Background = most common color
    bg = Counter(
        grid[r][c] for r in range(rows) for c in range(cols)
    ).most_common(1)[0][0]

    # 8-connected flood fill to find objects
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    def flood_fill(sr, sc):
        stack = [(sr, sc)]
        cells = []
        while stack:
            r, c = stack.pop()
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if visited[r][c] or grid[r][c] == bg:
                continue
            visited[r][c] = True
            cells.append((r, c, grid[r][c]))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        stack.append((r + dr, c + dc))
        return cells

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                cells = flood_fill(r, c)
                if cells:
                    objects.append(cells)

    arm_color = 1

    # Compute arm bounding box and scale for each object
    obj_info = []
    for cells in objects:
        arms = [(r, c) for r, c, v in cells if v == arm_color]
        if not arms:
            continue
        ar0 = min(r for r, c in arms)
        ac0 = min(c for r, c in arms)
        ar1 = max(r for r, c in arms)
        ac1 = max(c for r, c in arms)
        obj_info.append({
            'cells': cells,
            'arm_origin': (ar0, ac0),
            'arm_h': ar1 - ar0 + 1,
            'arm_w': ac1 - ac0 + 1,
        })

    min_h = min(o['arm_h'] for o in obj_info)
    for o in obj_info:
        o['scale'] = o['arm_h'] // min_h

    # Normalize templates using floor division by scale
    for o in obj_info:
        s = o['scale']
        r0, c0 = o['arm_origin']
        tmpl = {}
        for r, c, v in o['cells']:
            tr, tc = (r - r0) // s, (c - c0) // s
            tmpl[(tr, tc)] = v
        o['template'] = tmpl
        o['variant'] = frozenset(
            (tr, tc, v) for (tr, tc), v in tmpl.items() if v != arm_color
        )

    # Group by variant — expect exactly 2
    groups = defaultdict(list)
    for i, o in enumerate(obj_info):
        groups[o['variant']].append(i)

    variants = list(groups.keys())
    assert len(variants) == 2, f"Expected 2 variants, got {len(variants)}"
    swap = {variants[0]: variants[1], variants[1]: variants[0]}

    # Apply swap: erase old non-arm cells, draw new variant
    result = copy.deepcopy(grid)
    for var, indices in groups.items():
        target = swap[var]
        for idx in indices:
            o = obj_info[idx]
            s = o['scale']
            r0, c0 = o['arm_origin']

            # Erase current non-arm cells
            for r, c, v in o['cells']:
                if v != arm_color:
                    result[r][c] = bg

            # Draw target variant (scaled)
            for tr, tc, v in target:
                for dr in range(s):
                    for dc in range(s):
                        nr, nc = r0 + tr * s + dr, c0 + tc * s + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = v

    return result


if __name__ == '__main__':
    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else '/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/c7f57c3e.json'
    )
    with open(path) as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task['train']):
        pred = transform(ex['input'])
        expected = ex['output']
        ok = pred == expected
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if pred[r][c] != expected[r][c]:
                        print(f"  Diff ({r},{c}): got {pred[r][c]}, expected {expected[r][c]}")

    for i, ex in enumerate(task['test']):
        pred = transform(ex['input'])
        if 'output' in ex:
            expected = ex['output']
            ok = pred == expected
            print(f"Test {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                for r in range(len(expected)):
                    for c in range(len(expected[0])):
                        if pred[r][c] != expected[r][c]:
                            print(f"  Diff ({r},{c}): got {pred[r][c]}, expected {expected[r][c]}")
        else:
            print(f"Test {i}: prediction generated")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'FAIL'}")


solve = transform
