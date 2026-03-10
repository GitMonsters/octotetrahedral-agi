"""
Solver for ARC-AGI puzzle fc7cae8d.

The input grid has a frame with 4 border edges (2 solid, 2 sparse) and an inner
rectangle bounded by two border lines whose colors match the solid outer borders.
The output is the inner rectangle content, rotated/reflected so that each inner
border line aligns with its corresponding outer border position.
"""

import json
from collections import Counter
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    H = len(grid)
    W = len(grid[0])

    # --- Step 1: Identify the two solid outer borders ---
    edges = {
        "top": [grid[0][c] for c in range(1, W - 1)],
        "bottom": [grid[H - 1][c] for c in range(1, W - 1)],
        "left": [grid[r][0] for r in range(1, H - 1)],
        "right": [grid[r][W - 1] for r in range(1, H - 1)],
    }
    solid_borders = {}
    for name, vals in edges.items():
        c = Counter(v for v in vals)
        mc, mc_count = c.most_common(1)[0]
        if mc != 0 and mc != 5 and mc_count / len(vals) > 0.8:
            solid_borders[name] = mc
    color_to_border = {v: k for k, v in solid_borders.items()}
    solid_colors = list(solid_borders.values())

    # --- Step 2: Find inner border lines (horizontal and vertical) ---
    def longest_contiguous(positions):
        if not positions:
            return []
        best_start, best_len = 0, 1
        cur_start, cur_len = 0, 1
        for i in range(1, len(positions)):
            if positions[i] == positions[i - 1] + 1:
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
                cur_start, cur_len = i, 1
        if cur_len > best_len:
            best_len, best_start = cur_len, cur_start
        return positions[best_start : best_start + best_len]

    h_borders = []
    for r in range(2, H - 2):
        for color in solid_colors:
            positions = sorted(c for c in range(W) if grid[r][c] == color)
            contig = longest_contiguous(positions)
            if len(contig) >= 4:
                h_borders.append((r, contig[0], contig[-1], color))

    v_borders = []
    for c in range(2, W - 2):
        for color in solid_colors:
            positions = sorted(r for r in range(H) if grid[r][c] == color)
            contig = longest_contiguous(positions)
            if len(contig) >= 4:
                v_borders.append((c, contig[0], contig[-1], color))

    h_border = max(h_borders, key=lambda b: b[2] - b[1])
    v_border = max(v_borders, key=lambda b: b[2] - b[1])

    h_row, h_col_start, h_col_end, h_color = h_border
    v_col, v_row_start, v_row_end, v_color = v_border

    # --- Step 3: Extract content area ---
    r_start, r_end = v_row_start, v_row_end
    c_start, c_end = h_col_start, h_col_end
    content = [row[c_start : c_end + 1] for row in grid[r_start : r_end + 1]]
    cH, cW = len(content), len(content[0])

    # --- Step 4: Determine rotation via edge mapping ---
    h_pos = "T" if h_row < r_start else "B"
    v_pos = "L" if v_col < c_start else "R"
    h_target = color_to_border[h_color][0].upper()
    v_target = color_to_border[v_color][0].upper()

    D4 = {
        "identity": {"T": "T", "R": "R", "B": "B", "L": "L"},
        "cw90": {"T": "R", "R": "B", "B": "L", "L": "T"},
        "rot180": {"T": "B", "R": "L", "B": "T", "L": "R"},
        "ccw90": {"T": "L", "R": "T", "B": "R", "L": "B"},
        "flipud": {"T": "B", "R": "R", "B": "T", "L": "L"},
        "fliplr": {"T": "T", "R": "L", "B": "B", "L": "R"},
        "transpose": {"T": "L", "R": "B", "B": "R", "L": "T"},
        "anti_transpose": {"T": "R", "R": "T", "B": "L", "L": "B"},
    }

    transform_name = None
    for name, mapping in D4.items():
        if mapping[h_pos] == h_target and mapping[v_pos] == v_target:
            transform_name = name
            break

    # --- Step 5: Apply the transformation ---
    def apply_transform(mat, name):
        h, w = len(mat), len(mat[0])
        if name == "identity":
            return [row[:] for row in mat]
        elif name == "cw90":
            return [[mat[h - 1 - c][r] for c in range(h)] for r in range(w)]
        elif name == "rot180":
            return [[mat[h - 1 - r][w - 1 - c] for c in range(w)] for r in range(h)]
        elif name == "ccw90":
            return [[mat[c][w - 1 - r] for c in range(h)] for r in range(w)]
        elif name == "flipud":
            return [row[:] for row in reversed(mat)]
        elif name == "fliplr":
            return [row[::-1] for row in mat]
        elif name == "transpose":
            return [[mat[r][c] for r in range(h)] for c in range(w)]
        elif name == "anti_transpose":
            return [[mat[h - 1 - c][w - 1 - r] for c in range(h)] for r in range(w)]

    return apply_transform(content, transform_name)


if __name__ == "__main__":
    import os

    task_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "dataset",
        "tasks",
        "fc7cae8d.json",
    )
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        ok = result == expected
        print(f"Train {i+1}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Got shape:      {len(result)}x{len(result[0])}")

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        expected = ex["output"]
        ok = result == expected
        print(f"Test  {i+1}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
