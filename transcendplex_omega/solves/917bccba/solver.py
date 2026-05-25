"""
ARC-AGI puzzle 917bccba solver.

Pattern: A grid contains a rectangular border (color A) and a cross/plus shape
(color B) passing through it. The cross interior is erased and its outer arms
are relocated so the vertical arm aligns with the rectangle's right edge and
the horizontal arm aligns with the rectangle's top edge.
"""

import json
import numpy as np
from pathlib import Path


def solve(grid: list[list[int]]) -> list[list[int]]:
    g = np.array(grid)
    rows, cols = g.shape

    # Find the two non-zero colors
    colors = sorted(set(g.flatten()) - {0})

    # Identify the rectangle color: its bounding-box top/bottom rows are fully filled
    rect_color = cross_color = None
    rect_top = rect_bot = rect_left = rect_right = 0
    for color in colors:
        coords = np.argwhere(g == color)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        if (np.all(g[r_min, c_min:c_max + 1] == color) and
                np.all(g[r_max, c_min:c_max + 1] == color)):
            rect_color = color
            rect_top, rect_bot = int(r_min), int(r_max)
            rect_left, rect_right = int(c_min), int(c_max)
            break

    cross_color = [c for c in colors if c != rect_color][0]

    # Build output: blank grid → rectangle border → relocated cross arms
    out = np.zeros_like(g)

    # Draw rectangle border
    out[rect_top, rect_left:rect_right + 1] = rect_color
    out[rect_bot, rect_left:rect_right + 1] = rect_color
    out[rect_top:rect_bot + 1, rect_left] = rect_color
    out[rect_top:rect_bot + 1, rect_right] = rect_color

    # Vertical cross arm → right edge column, only outside the rectangle
    for r in range(rows):
        if r < rect_top or r > rect_bot:
            out[r, rect_right] = cross_color

    # Horizontal cross arm → top edge row, only outside the rectangle
    for c in range(cols):
        if c < rect_left or c > rect_right:
            out[rect_top, c] = cross_color

    return out.tolist()


# --------------- self-test ---------------
if __name__ == "__main__":
    task_path = Path("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/917bccba.json")
    task = json.load(open(task_path))

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        ok = result == expected
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            for r in range(len(expected)):
                if result[r] != expected[r]:
                    print(f"  Row {r}: got {result[r]}")
                    print(f"       exp {expected[r]}")

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            ok = result == ex["output"]
            print(f"Test  {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
        else:
            print(f"Test  {i}: (no expected output) predicted:")
            for row in result:
                print(f"  {row}")

    if all_pass:
        print("\n✅ All training examples PASSED")
    else:
        print("\n❌ Some examples FAILED")
