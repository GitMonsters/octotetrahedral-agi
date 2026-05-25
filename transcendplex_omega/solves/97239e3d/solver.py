"""
ARC-AGI task 97239e3d solver.

Pattern: 17x17 grid with 4x4 arrangement of 3x3 boxes (8-rings with 0-center),
separated by 0-valued grid lines at rows/cols 0,4,8,12,16.

Colored markers (non-0, non-8) appear on grid lines or inside boxes.
Same-color markers define a grid-aligned rectangle. The solver:
1. Finds the smallest grid-aligned rectangle containing all markers of each color
2. Draws the rectangle border with that color
3. Fills box centers inside the rectangle with that color
"""

def solve(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    step = 4  # grid line spacing
    out = [row[:] for row in grid]

    # Collect markers by color
    markers: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v not in (0, 8):
                markers.setdefault(v, []).append((r, c))

    for color, pts in markers.items():
        # Compute grid-aligned bounding rectangle
        top, bottom, left, right = H, 0, W, 0
        for r, c in pts:
            r_lo = (r // step) * step
            r_hi = r_lo + step if r % step != 0 else r_lo
            c_lo = (c // step) * step
            c_hi = c_lo + step if c % step != 0 else c_lo
            top = min(top, r_lo)
            bottom = max(bottom, r_hi)
            left = min(left, c_lo)
            right = max(right, c_hi)

        # Draw rectangle border
        for c in range(left, right + 1):
            out[top][c] = color
            out[bottom][c] = color
        for r in range(top, bottom + 1):
            out[r][left] = color
            out[r][right] = color

        # Fill box centers inside the rectangle
        for ci in range(H // step):
            for cj in range(W // step):
                ct = ci * step
                cb = (ci + 1) * step
                cl = cj * step
                cr = (cj + 1) * step
                if ct >= top and cb <= bottom and cl >= left and cr <= right:
                    out[ci * step + 2][cj * step + 2] = color

    return out


if __name__ == "__main__":
    import json, sys

    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/97239e3d.json"))

    ok = True
    for i, pair in enumerate(task["train"]):
        result = solve(pair["input"])
        match = result == pair["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            ok = False
            for r in range(len(result)):
                if result[r] != pair["output"][r]:
                    print(f"  row {r}: got  {result[r]}")
                    print(f"  row {r}: want {pair['output'][r]}")

    for i, pair in enumerate(task["test"]):
        result = solve(pair["input"])
        if "output" in pair:
            match = result == pair["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                ok = False
        else:
            print(f"Test  {i}: (no expected output)")
            for row in result:
                print(row)

    print(f"\nAll passed: {ok}")
