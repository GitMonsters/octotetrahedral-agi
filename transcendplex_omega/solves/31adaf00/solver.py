"""
ARC-AGI solver for task 31adaf00

Rule:
- The grid contains only 0s and 5s.
- Find all maximal rectangles of 0s that are also squares (NxN, N>=2).
  A maximal rectangle cannot be extended by any row or column while staying all-0.
- Exclude any maximal square that overlaps with a strictly larger maximal square.
- Fill the remaining squares with 1.
"""


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Prefix sums for fast all-zero rectangle queries
    psum = [[0] * (cols + 1) for _ in range(rows + 1)]
    for r in range(rows):
        for c in range(cols):
            psum[r + 1][c + 1] = (
                psum[r][c + 1] + psum[r + 1][c] - psum[r][c] + (0 if grid[r][c] == 0 else 1)
            )

    def all_zero(r1, c1, r2, c2):
        return (psum[r2 + 1][c2 + 1] - psum[r1][c2 + 1] - psum[r2 + 1][c1] + psum[r1][c1]) == 0

    def is_maximal(r1, c1, r2, c2):
        if r1 > 0 and all_zero(r1 - 1, c1, r1 - 1, c2):
            return False
        if r2 < rows - 1 and all_zero(r2 + 1, c1, r2 + 1, c2):
            return False
        if c1 > 0 and all_zero(r1, c1 - 1, r2, c1 - 1):
            return False
        if c2 < cols - 1 and all_zero(r1, c2 + 1, r2, c2 + 1):
            return False
        return True

    # Collect all maximal squares of size >= 2
    maximal_squares = []
    for r1 in range(rows):
        for c1 in range(cols):
            max_size = min(rows - r1, cols - c1)
            for size in range(2, max_size + 1):
                r2 = r1 + size - 1
                c2 = c1 + size - 1
                if all_zero(r1, c1, r2, c2) and is_maximal(r1, c1, r2, c2):
                    maximal_squares.append((r1, c1, size))

    def overlaps(a, b):
        ar, ac, az = a
        br, bc, bz = b
        return not (ar + az <= br or br + bz <= ar or ac + az <= bc or bc + bz <= ac)

    # Fill squares that don't overlap with any strictly larger maximal square
    for sq in maximal_squares:
        r, c, sz = sq
        has_larger = any(o[2] > sz and overlaps(sq, o) for o in maximal_squares)
        if not has_larger:
            for dr in range(sz):
                for dc in range(sz):
                    out[r + dr][c + dc] = 1

    return out


if __name__ == "__main__":
    import json, pathlib

    task_path = pathlib.Path(__file__).resolve().parents[2] / "dataset" / "tasks" / "31adaf00.json"
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for phase in ("train", "test"):
        for i, ex in enumerate(task[phase]):
            result = solve(ex["input"])
            ok = result == ex["output"]
            print(f"{phase}[{i}]: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                for r, (got, exp) in enumerate(zip(result, ex["output"])):
                    if got != exp:
                        print(f"  row {r}: got {got}")
                        print(f"          exp {exp}")
    raise SystemExit(0 if all_pass else 1)
