"""
ARC-AGI puzzle da515329 — Spiral/maze pattern solver.

Input: Grid with a cross/plus of 8s (center=0, arms=8) of arm length L.
Output: Concentric rectangular spiral of 8s and 0s radiating from the cross.

Pattern rules (Chebyshev-distance rings from center):
  d=0: center cell is 0.
  d < max(2,L): alternating checkerboard — value=8 iff position_parity == d_parity.
  d >= max(2,L): spiral zone — each ring is mostly one value (base) with
      minority "gaps" at specific positions within each quadrant of 2d elements.
      The gap offsets follow a fixed formula based on L.
"""

import json
import sys


def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Locate the cross: find center (cy, cx) and arm length L
    cells8 = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 8]
    rows8 = [r for r, c in cells8]
    cols8 = [c for r, c in cells8]
    cy = (min(rows8) + max(rows8)) // 2
    cx = (min(cols8) + max(cols8)) // 2
    L = cy - min(rows8)

    S = max(2, L)                   # first spiral-zone ring distance
    gap_parity = (S + 1) % 2       # parity of minority-offset positions
    n_gaps = max(1, L - 1)         # minority positions per quadrant
    n_front = (L + 1) // 2         # ceil(L/2): gaps taken from front
    n_back = n_gaps - n_front      # gaps taken from back

    out = [[0] * W for _ in range(H)]

    for r in range(H):
        for c in range(W):
            dr = r - cy
            dc = c - cx
            d = max(abs(dr), abs(dc))

            if d == 0:
                out[r][c] = 0
                continue

            # Position along ring d (clockwise from top-left corner)
            if dr == -d:
                p = dc + d
            elif dc == d:
                p = 3 * d + dr
            elif dr == d:
                p = 5 * d - dc
            else:
                p = 7 * d - dr

            if d < S:
                # Alternating zone: checkerboard matching d's parity
                out[r][c] = 8 if p % 2 == d % 2 else 0
            else:
                # Spiral zone
                quad_offset = p % (2 * d)

                is_gap = False
                if quad_offset % 2 == gap_parity:
                    # Index among same-parity offsets within the quadrant
                    idx = quad_offset // 2 if gap_parity == 0 else (quad_offset - 1) // 2
                    if idx < n_front or idx >= d - n_back:
                        is_gap = True

                if (d - S) % 2 == 0:
                    out[r][c] = 0 if is_gap else 8   # base=8, minority=0
                else:
                    out[r][c] = 8 if is_gap else 0   # base=0, minority=8

    return out


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI-2/data/evaluation/da515329.json"
    )
    with open(path) as f:
        data = json.load(f)

    all_pass = True
    for section in ["train", "test"]:
        for i, pair in enumerate(data.get(section, [])):
            result = transform(pair["input"])
            expected = pair.get("output")
            if expected is None:
                print(f"{section}[{i}]: no expected output — printing prediction")
                for row in result:
                    print(row)
                continue
            match = result == expected
            print(f"{section}[{i}]: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
                H = len(expected)
                W = len(expected[0])
                diffs = sum(
                    1
                    for r in range(H)
                    for c in range(W)
                    if result[r][c] != expected[r][c]
                )
                print(f"  {diffs} differences out of {H * W}")
                # Show first few diffs
                count = 0
                for r in range(H):
                    for c in range(W):
                        if result[r][c] != expected[r][c]:
                            print(f"  ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")
                            count += 1
                            if count >= 10:
                                break
                    if count >= 10:
                        break

    if all_pass:
        print("\nAll PASS!")


solve = transform
