"""
Solver for ARC-AGI task 4e45f183

The 19x19 grid contains a 3x3 arrangement of 5x5 sub-grids separated by
zero-borders. Each sub-grid has a pattern of "marker" cells (minority color)
on a background (majority color). One sub-grid holds the full/master pattern;
the other eight hold directional fragments.

The transformation rearranges sub-grids so each one sits at the position
indicated by the centroid of its marker cells within the 5x5 space:
  centroid < 1.5 → position 0, 1.5–2.5 → position 1, > 2.5 → position 2
"""

import json
from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])

    # Find border rows/cols (all zeros)
    zero_rows = [r for r in range(H) if all(grid[r][c] == 0 for c in range(W))]
    zero_cols = [c for c in range(W) if all(grid[r][c] == 0 for r in range(H))]

    # Derive sub-grid ranges between borders
    row_ranges = []
    for i in range(len(zero_rows) - 1):
        s, e = zero_rows[i] + 1, zero_rows[i + 1]
        if s < e:
            row_ranges.append((s, e))

    col_ranges = []
    for i in range(len(zero_cols) - 1):
        s, e = zero_cols[i] + 1, zero_cols[i + 1]
        if s < e:
            col_ranges.append((s, e))

    sg_h = row_ranges[0][1] - row_ranges[0][0]
    sg_w = col_ranges[0][1] - col_ranges[0][0]

    # Extract each sub-grid
    sub_grids: dict[tuple[int, int], list[list[int]]] = {}
    for gi, (rs, re) in enumerate(row_ranges):
        for gj, (cs, ce) in enumerate(col_ranges):
            sub_grids[(gi, gj)] = [
                [grid[r][c] for c in range(cs, ce)] for r in range(rs, re)
            ]

    # Identify pattern color (minority) vs background (majority)
    color_counts: Counter = Counter()
    for sg in sub_grids.values():
        for row in sg:
            color_counts.update(row)
    bg_color = color_counts.most_common(1)[0][0]
    pattern_color = [c for c in color_counts if c != bg_color][0]

    center_r = (sg_h - 1) / 2.0
    center_c = (sg_w - 1) / 2.0

    # Classify each sub-grid by centroid of its pattern cells
    target_map: dict[tuple[int, int], list[list[int]]] = {}
    for (gi, gj), sg in sub_grids.items():
        positions = [
            (r, c)
            for r in range(sg_h)
            for c in range(sg_w)
            if sg[r][c] == pattern_color
        ]
        if not positions:
            continue

        avg_r = sum(p[0] for p in positions) / len(positions)
        avg_c = sum(p[1] for p in positions) / len(positions)

        tr = 0 if avg_r < center_r - 0.5 else (2 if avg_r > center_r + 0.5 else 1)
        tc = 0 if avg_c < center_c - 0.5 else (2 if avg_c > center_c + 0.5 else 1)

        target_map[(tr, tc)] = sg

    # Build output: copy borders, then place sub-grids at target positions
    output = [row[:] for row in grid]
    for (ti, tj), sg in target_map.items():
        rs = row_ranges[ti][0]
        cs = col_ranges[tj][0]
        for r in range(sg_h):
            for c in range(sg_w):
                output[rs + r][cs + c] = sg[r][c]

    return output


if __name__ == "__main__":
    with open(
        "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/4e45f183.json"
    ) as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Training example {i}: PASS")
        else:
            all_pass = False
            print(f"Training example {i}: FAIL")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(
                            f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}"
                        )

    if all_pass:
        print("\nAll training examples passed!")
    else:
        print("\nSome examples failed.")
