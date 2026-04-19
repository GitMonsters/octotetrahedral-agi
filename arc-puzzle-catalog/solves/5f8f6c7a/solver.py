"""
ARC-AGI Task 5f8f6c7a Solver

Rule: Each marker (non-background cell) creates two staircase beams colored 6:
  - DOWN staircase (goes down-right):
      odd step j:  isolated cell at (r0+j, c0+j-1)
      even step j: 3-cell block at (r0+j, c0+j-2), (r0+j, c0+j-1), (r0+j, c0+j)
  - UP staircase (goes up-left):
      odd step j:  isolated cell at (r0-j, c0-j+1)
      even step j: 3-cell block at (r0-j, c0-j), (r0-j, c0-j+1), (r0-j, c0-j+2)

The output color is always 6. Staircase cells never overwrite markers.
"""
import copy
from collections import Counter


def transform(grid):
    H = len(grid)
    W = len(grid[0])

    # Determine background (most frequent color)
    freq = Counter(v for row in grid for v in row)
    bg = freq.most_common(1)[0][0]

    # Find all marker positions (non-background cells)
    markers = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
    marker_set = set(markers)

    out_color = 6
    out = copy.deepcopy(grid)

    def paint(r, c):
        if 0 <= r < H and 0 <= c < W and (r, c) not in marker_set:
            out[r][c] = out_color

    for r0, c0 in markers:
        # DOWN staircase: diagonal beam going down-right
        for j in range(1, H + W):
            r = r0 + j
            if r >= H:
                break
            if j % 2 == 1:  # odd step: single isolated cell
                paint(r, c0 + j - 1)
            else:            # even step: 3-cell horizontal block
                paint(r, c0 + j - 2)
                paint(r, c0 + j - 1)
                paint(r, c0 + j)

        # UP staircase: diagonal beam going up-left
        for j in range(1, H + W):
            r = r0 - j
            if r < 0:
                break
            if j % 2 == 1:  # odd step: single isolated cell
                paint(r, c0 - j + 1)
            else:            # even step: 3-cell horizontal block
                paint(r, c0 - j)
                paint(r, c0 - j + 1)
                paint(r, c0 - j + 2)

    return out


if __name__ == "__main__":
    import json

    with open("/Users/evanpieser/Desktop/ReArc45/re-arc_test_challenges-2026-04-05T23-26-25.json") as f:
        data = json.load(f)

    task = data["5f8f6c7a"]
    all_pass = True
    for i, pair in enumerate(task["train"]):
        predicted = transform(pair["input"])
        expected = pair["output"]
        passed = predicted == expected
        print(f"Train pair {i}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False
            H, W = len(expected), len(expected[0])
            diffs = [
                (r, c, expected[r][c], predicted[r][c])
                for r in range(H)
                for c in range(W)
                if expected[r][c] != predicted[r][c]
            ]
            print(f"  First diffs: {diffs[:10]}")

    print(f"\nAll training pairs passed: {all_pass}")
