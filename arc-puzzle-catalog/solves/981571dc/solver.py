"""
Solver for ARC task 981571dc.

The output grid is a symmetric matrix with row palindromes from position 2:
  1. Transpose symmetry: out[r][c] == out[c][r]
  2. Row palindrome from position 2: out[r][c] == out[r][N+1-c] for c >= 2
     where N = number of columns - 1.

These two symmetries generate equivalence classes of up to 8 positions.
For each zero cell, we find a non-zero equivalent position and copy its value.
"""
import json


def solve(grid):
    R = len(grid)
    C = len(grid[0])
    out = [row[:] for row in grid]

    def get_equivalents(r, c):
        positions = {(r, c), (c, r)}
        if c >= 2:
            mc = R + 1 - c
            if 0 <= mc < C:
                positions.add((r, mc))
                positions.add((mc, r))
        if r >= 2:
            mr = C + 1 - r
            if 0 <= mr < R:
                positions.add((c, mr))
                positions.add((mr, c))
        if r >= 2 and c >= 2:
            mc = R + 1 - c
            mr = C + 1 - r
            if 0 <= mc < C and 0 <= mr < R:
                positions.add((mr, mc))
                positions.add((mc, mr))
        return positions

    changed = True
    while changed:
        changed = False
        for r in range(R):
            for c in range(C):
                if out[r][c] == 0:
                    for er, ec in get_equivalents(r, c):
                        if 0 <= er < R and 0 <= ec < C and out[er][ec] != 0:
                            out[r][c] = out[er][ec]
                            changed = True
                            break
    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/981571dc.json") as f:
        data = json.load(f)

    all_pass = True
    for i, ex in enumerate(data["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            R, C = len(expected), len(expected[0])
            diffs = sum(
                1 for r in range(R) for c in range(C) if result[r][c] != expected[r][c]
            )
            print(f"Train {i}: FAIL ({diffs} mismatches)")
            all_pass = False

    for i, ex in enumerate(data["test"]):
        if "output" in ex:
            result = solve(ex["input"])
            expected = ex["output"]
            if result == expected:
                print(f"Test  {i}: PASS")
            else:
                R, C = len(expected), len(expected[0])
                diffs = sum(
                    1
                    for r in range(R)
                    for c in range(C)
                    if result[r][c] != expected[r][c]
                )
                print(f"Test  {i}: FAIL ({diffs} mismatches)")
                all_pass = False
        else:
            result = solve(ex["input"])
            print(f"Test  {i}: produced {len(result)}x{len(result[0])} grid")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")
