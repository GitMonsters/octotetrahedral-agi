"""Solver for 221dfab4 — Marker beam projection through shapes

A marker bar on one edge fires a beam perpendicular into the grid.
The beam repeats a period-6 pattern (marker, bg, marker, bg, green, bg).
On green-distance slices, all shape cells in that row/column also turn green.
"""
import json
from typing import List
from collections import Counter


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])

    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    non_bg = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    color_counts = Counter(v for _, _, v in non_bg)
    marker_color = min(color_counts, key=color_counts.get)
    shape_color = max(color_counts, key=color_counts.get)

    marker_cells = [(r, c) for r, c, v in non_bg if v == marker_color]
    shape_set = set((r, c) for r, c, v in non_bg if v == shape_color)

    output = [row[:] for row in grid]

    mr_set = set(r for r, _ in marker_cells)
    mc_set = set(c for _, c in marker_cells)

    if len(mr_set) == 1:
        # Horizontal marker → vertical beam
        mr = next(iter(mr_set))
        c_lo, c_hi = min(mc_set), max(mc_set)
        step = -1 if mr > rows // 2 else 1

        d, r = 0, mr
        while 0 <= r < rows:
            phase = d % 6
            beam_val = marker_color if phase in (0, 2) else (3 if phase == 4 else bg)
            for c in range(c_lo, c_hi + 1):
                output[r][c] = beam_val
            if phase == 4:
                for c in range(cols):
                    if (r, c) in shape_set:
                        output[r][c] = 3
            d += 1
            r = mr + step * d
    else:
        # Vertical marker → horizontal beam
        mc = next(iter(mc_set))
        r_lo, r_hi = min(mr_set), max(mr_set)
        step = -1 if mc > cols // 2 else 1

        d, c = 0, mc
        while 0 <= c < cols:
            phase = d % 6
            beam_val = marker_color if phase in (0, 2) else (3 if phase == 4 else bg)
            for r in range(r_lo, r_hi + 1):
                output[r][c] = beam_val
            if phase == 4:
                for r in range(rows):
                    if (r, c) in shape_set:
                        output[r][c] = 3
            d += 1
            c = mc + step * d

    return output


if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f: task = json.load(f)
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            result = solve(ex['input'])
            match = result == ex['output']
            if not match:
                diffs = sum(1 for r in range(len(result)) for c in range(len(result[0]))
                            if result[r][c] != ex['output'][r][c])
                print(f"{split.title()} {i}: FAIL ({diffs} diffs)")
            else:
                print(f"{split.title()} {i}: PASS ✓")
