"""
ARC-AGI task 93c31fbe solver.

Pattern: Rectangular frames made of 4 L-shaped corners (non-0, non-1 color).
Blue (1) pixels inside frames are reflected across the frame's symmetry axis:
  - Wider frames → horizontal (left-right) reflection
  - Taller frames → vertical (top-bottom) reflection
Blue pixels outside any frame are removed.
"""

import json
import copy


def solve(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    out = copy.deepcopy(grid)

    # Clear all blue pixels; we'll add correct ones back
    for r in range(H):
        for c in range(W):
            if out[r][c] == 1:
                out[r][c] = 0

    # Find frame colors
    frame_colors = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] not in (0, 1):
                frame_colors.add(grid[r][c])

    frames = []

    for C in frame_colors:
        tl, tr, bl, br = [], [], [], []

        for r in range(H - 1):
            for c in range(W - 1):
                a, b = grid[r][c], grid[r][c + 1]
                d, e = grid[r + 1][c], grid[r + 1][c + 1]

                # TL corner: CC / C.
                if a == C and b == C and d == C and e != C:
                    tl.append((r, c))          # frame top-row, left-col
                # TR corner: CC / .C
                if a == C and b == C and d != C and e == C:
                    tr.append((r, c + 1))      # frame top-row, right-col
                # BL corner: C. / CC
                if a == C and b != C and d == C and e == C:
                    bl.append((r + 1, c))      # frame bottom-row, left-col
                # BR corner: .C / CC
                if a != C and b == C and d == C and e == C:
                    br.append((r + 1, c + 1))  # frame bottom-row, right-col

        br_set = set(br)
        for (top, left) in tl:
            for (top2, right) in tr:
                if top2 != top:
                    continue
                for (bottom, left2) in bl:
                    if left2 != left:
                        continue
                    if (bottom, right) in br_set:
                        frames.append((top, left, bottom, right))

    # Process each frame
    for (top, left, bottom, right) in frames:
        w = right - left + 1
        h = bottom - top + 1

        # Collect blue pixels inside the frame bounding box
        blues = []
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if grid[r][c] == 1:
                    blues.append((r, c))

        # Place originals
        for (r, c) in blues:
            out[r][c] = 1

        # Reflect and place copies
        if w > h:
            # Wider → reflect left-right
            center_col = (left + right) / 2.0
            for (r, c) in blues:
                nc = int(round(2 * center_col - c))
                if left <= nc <= right:
                    out[r][nc] = 1
        else:
            # Taller (or square) → reflect top-bottom
            center_row = (top + bottom) / 2.0
            for (r, c) in blues:
                nr = int(round(2 * center_row - r))
                if top <= nr <= bottom:
                    out[nr][c] = 1

    return out


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/93c31fbe.json"))

    all_pass = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        ok = result == pair["output"]
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            for r in range(len(result)):
                if result[r] != pair["output"][r]:
                    print(f"  row {r}: got  {result[r]}")
                    print(f"         want {pair['output'][r]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            ok = result == pair["output"]
            print(f"Test  {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
        else:
            print(f"Test  {i}: (no expected output) produced {len(result)}x{len(result[0])}")

    print(f"\nAll pass: {all_pass}")
